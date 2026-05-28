from typing import Dict, Literal

import cv2
import numpy as np
import pyvista as pv
import trimesh
import warp as wp
from chainner_ext import ResizeFilter, resize
from matplotlib import cm

wp.config.quiet = True


def itk_monkey_patch():
    import locale

    import itk

    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

    # 修正 itk.imread 读取 Direction 错误，导致 TotalSeg 分割错误
    # itk 默认 ForceOrthogonalDirection=True 不适用于从脚到头 (FFS) 或斜切 (gantry-tilt) 扫描
    _New = itk.ImageSeriesReader.New
    itk.ImageSeriesReader.New = lambda *a, **k: _New(*a, **{**k, 'ForceOrthogonalDirection': False})


itk_monkey_patch()


@wp.kernel
def _wp_closest_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    out_closest: wp.array(dtype=wp.vec3),
    out_dist: wp.array(dtype=wp.float32),
    out_tid: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    p = points[tid]

    # 执行 Warp 内置的 Mesh 查询
    # max_dist 设为极大值以模拟无限制查询
    query = wp.mesh_query_point(mesh_id, p, 1e10)

    if query.result:
        # 计算查询点到最近点的向量
        closest_p = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out_closest[tid] = closest_p
        out_dist[tid] = wp.length(p - closest_p)
        out_tid[tid] = query.face
    else:
        out_tid[tid] = -1


def closest_point(mesh, points):
    """
    使用 NVIDIA Warp 替换 trimesh.proximity.closest_point
    """
    points = np.asanyarray(points, dtype=np.float32)  # Warp 常用 float32
    num_points = len(points)

    # 1. 缓存 Warp Mesh 对象，避免重复构建 BVH
    if not hasattr(mesh, '_warp_mesh') or mesh._warp_mesh is None:
        # 将 trimesh 数据传输到 Warp 数组
        faces = wp.array(mesh.faces.flatten(), dtype=wp.int32)
        verts = wp.array(mesh.vertices, dtype=wp.vec3)

        # 创建 Warp Mesh 句柄
        mesh_id = wp.Mesh(verts, faces)
        mesh._warp_mesh = mesh_id
    else:
        mesh_id = mesh._warp_mesh

    # 2. 准备输入/输出数据 (GPU Array)
    wp_points = wp.from_numpy(points, dtype=wp.vec3)
    out_closest = wp.zeros(num_points, dtype=wp.vec3)
    out_dist = wp.zeros(num_points, dtype=wp.float32)
    out_tid = wp.zeros(num_points, dtype=wp.int32)

    # 3. 启动 Kernel
    wp.launch(
        kernel=_wp_closest_point_kernel,
        dim=num_points,
        inputs=[mesh_id.id, wp_points, out_closest, out_dist, out_tid],
    )

    # 4. 将结果转回 NumPy 并保持接口一致
    return (
        out_closest.numpy().astype(np.float64),
        out_dist.numpy().astype(np.float64),
        out_tid.numpy().astype(np.int32),
    )


def trimesh_monkey_patch():
    trimesh.proximity.closest_point = closest_point


def tri_poly(tri: trimesh.Trimesh | tuple, cell_data: Dict[str, np.ndarray] = None):
    if isinstance(tri, trimesh.Trimesh):
        tri = (tri.vertices, tri.faces)
    _ = np.insert(tri[1].reshape(-1, 3), 0, 3, axis=1)

    pd = pv.PolyData(tri[0], _)

    if cell_data is not None:
        for name, data in cell_data.items():
            pd.cell_data[name] = data

    return pd


def diff_dmc(
    volume: wp.array3d(dtype=wp.float32),
    origin: np.ndarray,
    spacing: float | np.ndarray,
    threshold: float,
):
    # vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
    #     -volume,
    #     -threshold,
    #     wp.vec3(origin),
    #     wp.vec3(origin + spacing * (np.array(volume.shape) - 1)),
    # )
    # return trimesh.Trimesh(vertices.numpy(), indices.numpy().reshape((-1, 3)))
    import torch
    from diso import DiffDMC

    vertices, indices = DiffDMC(dtype=torch.float32)(-wp.to_torch(volume), None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()
    vertices = vertices * spacing * (np.array(volume.shape) - 1) + origin
    return trimesh.Trimesh(vertices, indices)


def icp(a, b, initial=None, threshold=1e-5, max_iterations=20, **kwargs):
    """
    Apply the iterative closest point algorithm to align a point cloud with
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    a : (n,3) float
      List of points in space.
    b : (m,3) float or Trimesh
      List of points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    kwargs : dict
      Args to pass to procrustes

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """
    from scipy.spatial import cKDTree
    from trimesh import util
    from trimesh.registration import procrustes
    from trimesh.transformations import transform_points

    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    if initial is None:
        initial = np.eye(4)

    is_mesh = util.is_instance_named(b, 'Trimesh')
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError('points must be (n,3)!')
        btree = cKDTree(b)

    # transform a under initial_transformation
    a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for it in range(max_iterations):
        # Closest point in b to each point in a
        if is_mesh:
            closest, _distance, _faces = b.nearest.on_surface(a)
        else:
            _distances, ix = btree.query(a, 1)
            closest = b[ix]

        # align a with closest points
        matrix, transformed, cost = procrustes(a=a, b=closest, **kwargs)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost, it


trimesh_monkey_patch()


def fast_drr(a, ax, th=(0, 900), mode: Literal['mean', 'max'] = 'mean'):
    a = a.copy()
    c = th[0] < a
    a *= c
    if mode == 'mean':
        a = a.sum(axis=ax)
        c = np.sum(c, axis=ax)
        c[np.where(c <= 0)] = 1
        a = a / c
    elif mode == 'max':
        a = a.max(axis=ax)

    sm = cm.ScalarMappable(cmap='grey')
    sm.set_clim(th)
    a = sm.to_rgba(a, bytes=True)

    return a[:, :, :3].copy()


def resize_uint8(img, shape):
    img = img.astype(np.float32) / 255.0
    img = resize(img, tuple(round(_) for _ in reversed(shape)), ResizeFilter.CubicMitchell, False)
    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img


def cv2_line(img, p0, p1, color, thickness=1):
    cv2.line(img, p0[::-1], p1[::-1], color, thickness)


def itk_threshold_largest(label, lower=1, upper=255):
    import itk

    if isinstance(label, np.ndarray):
        inp = itk.image_from_array(label)
    else:
        inp = label

    threshold = itk.binary_threshold_image_filter(
        inp,
        lower_threshold=lower,
        upper_threshold=upper,
        inside_value=1,
        outside_value=0,
    )
    connected = itk.connected_component_image_filter(threshold)
    largest = itk.label_shape_keep_n_objects_image_filter(connected, background_value=0, number_of_objects=1, attribute='NumberOfPixels')
    out = itk.binary_threshold_image_filter(
        largest,
        lower_threshold=1,
        upper_threshold=2**31 - 1,
        inside_value=1,
        outside_value=0,
    )

    if isinstance(label, np.ndarray):
        out = itk.array_from_image(out)

    return out


@wp.kernel
def compute_sdf(
    mesh: wp.uint64,
    vertices: wp.array1d(dtype=wp.vec3),
    sdf: wp.array1d(dtype=float),
    max_dist: wp.float32,
):
    i = wp.tid()

    p = vertices[i]
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    dxyz = p - closest
    d = q.sign * wp.length(dxyz)
    sdf[i] = d


@wp.kernel
def resample_ab(
    image_b: wp.array3d(dtype=float),
    xform_a: wp.transform,
    volume_b: wp.uint64,
    roi_offset: wp.vec3,
    origin_a: wp.vec3,
    origin_b: wp.vec3,
    spacing_a: wp.vec3,
    spacing_b: wp.vec3,
):
    i, j, k = wp.tid()

    pa = wp.cw_mul(spacing_a, roi_offset + wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))) + origin_a
    pb = wp.transform_point(xform_a, pa)

    uvw = wp.cw_div(pb - origin_b, spacing_b)
    b = wp.volume_sample_f(volume_b, uvw, wp.Volume.LINEAR)

    image_b[i, j, k] = b


@wp.func
def bone_normalize(ct_value: float) -> float:
    if 150.0 <= ct_value < 650.0:
        value = -1.0 + (ct_value - 150.0) / 500.0 * 1.0
    elif 650.0 <= ct_value < 1150.0:
        value = 0.0 + (ct_value - 650.0) / 500.0 * 0.5
    elif 1150.0 <= ct_value < 3150.0:
        value = 0.5 + (ct_value - 1150.0) / 2000.0 * 0.5
    elif ct_value >= 3150.0:
        value = 1.0
    else:
        value = -1.0
    return value


@wp.kernel
def resample_roi(
    roi_images: wp.array3d(dtype=wp.vec3),
    roi_origin: wp.vec3,
    roi_spacing: wp.vec3,
    roi_hip_min: wp.vec3,
    roi_hip_max: wp.vec3,
    roi_fem_min: wp.vec3,
    roi_fem_max: wp.vec3,
    xform_hip: wp.transform,
    xform_fem: wp.transform,
    pre_volume: wp.uint64,
    pre_origin: wp.vec3,
    pre_spacing: wp.vec3,
    post_volume: wp.uint64,
    post_origin: wp.vec3,
    post_spacing: wp.vec3,
    hip_metal: wp.array3d(dtype=wp.float32),
    fem_metal: wp.array3d(dtype=wp.float32),
    ct_metal: wp.float32,
):
    i, j, k = wp.tid()

    pos = roi_origin + wp.cw_mul(roi_spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    uvw = wp.cw_div(pos - pre_origin, pre_spacing)
    pix = wp.volume_sample_f(pre_volume, uvw, wp.Volume.LINEAR)

    hip_pos = wp.transform_point(xform_hip, pos)
    hip_uvw = wp.cw_div(hip_pos - post_origin, post_spacing)
    hip_pix = wp.volume_sample_f(post_volume, hip_uvw, wp.Volume.LINEAR)

    fem_pos = wp.transform_point(xform_fem, pos)
    fem_uvw = wp.cw_div(fem_pos - post_origin, post_spacing)
    fem_pix = wp.volume_sample_f(post_volume, fem_uvw, wp.Volume.LINEAR)

    hip_in = roi_hip_min[0] <= pos[0] <= roi_hip_max[0] and roi_hip_min[1] <= pos[1] <= roi_hip_max[1] and roi_hip_min[2] <= pos[2] <= roi_hip_max[2]

    fem_in = roi_fem_min[0] <= pos[0] <= roi_fem_max[0] and roi_fem_min[1] <= pos[1] <= roi_fem_max[1] and roi_fem_min[2] <= pos[2] <= roi_fem_max[2]

    roi_images[i, j, k] = wp.vec3(bone_normalize(pix), bone_normalize(hip_pix), bone_normalize(fem_pix))

    if hip_in:
        if hip_pix >= ct_metal:
            hip_metal[i, j, k] = 1.0

    if fem_in:
        if fem_pix >= ct_metal:
            fem_metal[i, j, k] = 1.0


@wp.kernel
def resample_metal(
    hip_image: wp.array3d(dtype=wp.float32),
    fem_image: wp.array3d(dtype=wp.float32),
    roi_origin: wp.vec3,
    roi_spacing: wp.vec3,
    hip_mesh: wp.uint64,
    fem_mesh: wp.uint64,
    hip_cup_center: wp.vec3,
    fem_cup_center: wp.vec3,
    hip_cup_axis: wp.vec3,
    fem_cup_axis: wp.vec3,
    hip_head_center: wp.vec3,
    fem_head_center: wp.vec3,
    cup_radius: wp.float32,
    head_radius: wp.float32,
    sdf_t: wp.float32,
    max_dist: wp.float32,
):
    i, j, k = wp.tid()

    pos = roi_origin + wp.cw_mul(roi_spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    fem_cup_axis = wp.normalize(fem_cup_axis)

    # 柄颈平面
    hip_neck_plane = wp.dot(pos - hip_head_center, hip_cup_axis) - head_radius
    fem_neck_plane = wp.dot(pos - fem_head_center, fem_cup_axis) - head_radius

    # 1. 髋臼侧假体组件（去柄）
    query_hip = wp.mesh_query_point_sign_winding_number(hip_mesh, pos, max_dist)
    if query_hip.result:
        closest_hip = wp.mesh_eval_position(hip_mesh, query_hip.face, query_hip.u, query_hip.v)
        d_hip = -query_hip.sign * wp.length(pos - closest_hip)
    else:
        d_hip = -sdf_t

    sdf_cup = wp.min(d_hip, -hip_neck_plane)

    # 2. 股骨侧假体组件
    query = wp.mesh_query_point_sign_winding_number(fem_mesh, pos, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(fem_mesh, query.face, query.u, query.v)
        d_fem = -query.sign * wp.length(pos - closest)
    else:
        d_fem = -sdf_t

    sdf_stem = wp.min(d_fem, fem_neck_plane)

    # 4. 移除了两侧的头球体融合，髋臼侧仅保留完美球壳 (sdf_cup)，股骨侧仅保留切除球头后的柄体 (sdf_stem)
    hip_final = wp.clamp(sdf_cup / sdf_t, -1.0, 1.0)
    fem_final = wp.clamp(sdf_stem / sdf_t, -1.0, 1.0)

    hip_image[i, j, k] = hip_final
    fem_image[i, j, k] = fem_final


@wp.kernel
def resample_cup_head(
    volume: wp.uint64,
    origin: wp.vec3,
    spacing: wp.vec3,
    ct_min: float,
    ct_width: float,
    roi_image: wp.array2d(dtype=wp.vec3ub),
    roi_origin: wp.vec3,
    roi_spacing: wp.float32,
    axis_i: wp.vec3,
    axis_j: wp.vec3,
    cup_center: wp.vec3,
    cup_axis: wp.vec3,
    head_center: wp.vec3,
    head_radius: wp.float32,
    cup_radius: wp.float32,
):
    i, j = wp.tid()

    p = roi_origin + axis_i * wp.float32(i) * roi_spacing + axis_j * wp.float32(j) * roi_spacing

    uvw = wp.vec3(wp.cw_div(p - origin, spacing))
    grey = float(wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR))

    grey = 255.0 * (grey - ct_min) / ct_width
    grey = wp.clamp(grey, 0.0, 255.0)

    r = grey
    g = grey
    b = grey

    to_p_cup = p - cup_center
    cup_r = wp.length(to_p_cup)
    dot_cup = wp.dot(to_p_cup, cup_axis)

    to_p_head = p - head_center
    head_r = wp.length(to_p_head)
    dot_head = wp.dot(to_p_head, cup_axis)

    # th = roi_spacing * 2.1

    in_head = dot_head <= 0.5 * head_radius and (head_r <= head_radius)
    # out_head = dot_head <= 0.5 * head_radius and (head_radius < head_r <= head_radius + th)

    # th = (cup_radius - head_radius) * 0.5

    in_cup = dot_cup <= 0.0 and (cup_radius * 0.5 + head_radius * 0.5 < cup_r <= cup_radius)
    # out_cup = dot_cup <= 0.0 and (cup_radius < cup_r <= cup_radius + th)
    # out_cup = out_cup or (dot_cup <= 0.5 * cup_radius and (cup_radius - th < cup_r <= cup_radius + th))

    # in_liner = dot_cup <= 0.0 and (head_radius < cup_r <= head_radius + th)

    if in_head:
        r, g, b = r * 0.5 + 127.0, g * 0.5, b * 0.5
    # elif out_head:
    #     r, g, b = r * 0.5 + 64.0, g * 0.5, b * 0.5
    elif in_cup:
        r, g, b = r * 0.5, g * 0.5 + 64.0, b * 0.5 + 127.0
    # elif out_cup:
    #     r, g, b = r * 0.5, g * 0.5 + 32.0, b * 0.5 + 64.0
    # elif in_liner:
    #     r, g, b = r * 0.5 + 64.0, g * 0.5 + 64.0, b * 0.5 + 64.0

    r = wp.clamp(r, 0.0, 255.0)
    g = wp.clamp(g, 0.0, 255.0)
    b = wp.clamp(b, 0.0, 255.0)

    roi_image[i, j] = wp.vec3ub(wp.uint8(r), wp.uint8(g), wp.uint8(b))


@wp.kernel
def count_cup_head_3d(
    volume: wp.uint64,
    origin: wp.vec3,
    spacing: wp.vec3,
    ct_highlight: float,
    roi_origin: wp.vec3,
    roi_spacing: wp.float32,
    cup_center: wp.vec3,
    cup_axis: wp.vec3,
    head_center: wp.vec3,
    head_radius: wp.float32,
    cup_radius: wp.float32,
    counts: wp.array1d(dtype=wp.int32),
    shell_only: bool,
):
    i, j, k = wp.tid()

    p = (
        roi_origin
        + wp.vec3(1.0, 0.0, 0.0) * wp.float32(i) * roi_spacing
        + wp.vec3(0.0, 1.0, 0.0) * wp.float32(j) * roi_spacing
        + wp.vec3(0.0, 0.0, 1.0) * wp.float32(k) * roi_spacing
    )

    uvw = wp.vec3(wp.cw_div(p - origin, spacing))
    grey = float(wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR))

    to_p_cup = p - cup_center
    cup_r = wp.length(to_p_cup)
    dot_cup = wp.dot(to_p_cup, cup_axis)

    to_p_head = p - head_center
    head_r = wp.length(to_p_head)
    dot_head = wp.dot(to_p_head, cup_axis)

    th = (cup_radius - head_radius) * 0.5
    in_cup = dot_cup <= 0.0 and (cup_radius - th < cup_r <= cup_radius)
    out_cup = dot_cup <= 0.0 and (cup_radius < cup_r <= cup_radius + th)
    out_cup = out_cup or (dot_cup <= 0.5 * cup_radius and (cup_radius - th < cup_r <= cup_radius + th))

    in_liner = dot_cup <= 0.0 and (head_radius < cup_r <= head_radius + th)

    if shell_only:
        th = roi_spacing * 2.1

        in_head = dot_head <= 0.5 * head_radius and (head_radius - th < head_r <= head_radius)
        out_head = dot_head <= 0.5 * head_radius and (head_radius < head_r <= head_radius + th)
    else:
        in_head = dot_head <= 0.5 * head_radius and (head_r <= head_radius)
        out_head = dot_head <= 0.5 * head_radius and (head_radius < head_r <= head_radius + th)

    if in_head:
        wp.atomic_add(counts, 0, 1)
        if ct_highlight < grey:
            wp.atomic_add(counts, 1, 1)
    elif out_head:
        wp.atomic_add(counts, 2, 1)
        if ct_highlight < grey:
            wp.atomic_add(counts, 3, 1)
    elif in_cup:
        wp.atomic_add(counts, 4, 1)
        if ct_highlight < grey:
            wp.atomic_add(counts, 5, 1)
    elif out_cup:
        wp.atomic_add(counts, 6, 1)
        if ct_highlight < grey:
            wp.atomic_add(counts, 7, 1)
    elif in_liner:
        wp.atomic_add(counts, 8, 1)
        if ct_highlight < grey:
            wp.atomic_add(counts, 9, 1)


@wp.kernel
def contact_drr(
    image_a: wp.array3d(dtype=float),
    image_b: wp.array3d(dtype=float),
    image_3: wp.array3d(dtype=float),
    image_2: wp.array2d(dtype=wp.vec3ub),
    ct_bone: float,
    ct_metal: float,
    ct_bg: float,
    ct_min: float,
    ct_width: float,
    ax: int,
    swap: bool,
    proximal: wp.vec3i,
    distal: wp.vec3i,
):
    i, j = wp.tid()

    end = image_a.shape[ax]
    add = float(0.0)
    count = float(0.0)
    count_ = wp.vec2()

    for k in range(end):
        if ax == 1:
            ijk = wp.vec3i(i, wp.int32(k), j)
            ax_ = 2
        elif ax == 2:
            ijk = wp.vec3i(i, j, wp.int32(k))
            ax_ = 1
        else:
            ijk = wp.vec3i(wp.int32(k), i, j)
            ax_ = 2

        _3 = image_3[ijk[0], ijk[1], ijk[2]]
        if _3 > ct_min:
            add += _3
            count += 1.0

        if ijk[0] > proximal[0]:
            continue

        a = image_a[ijk[0], ijk[1], ijk[2]]
        b = image_b[ijk[0], ijk[1], ijk[2]]

        if swap:
            a, b = b, a

        if a >= ct_bone and b < ct_metal:  # 原位是骨
            for _ in range(2):
                ijk_ = ijk
                ijk_[ax_] += 1 if _ > 0 else -1
                if ijk_[ax_] < 0 or ijk_[ax_] >= image_a.shape[ax_]:
                    continue

                a_ = image_a[ijk_[0], ijk_[1], ijk_[2]]
                b_ = image_b[ijk_[0], ijk_[1], ijk_[2]]

                if swap:
                    a_, b_ = b_, a_

                if b_ >= ct_metal:  # 邻位有金属
                    count_[_] += 1.0

    if count > 0.0:
        grey = add / count
    else:
        grey = ct_bg

    grey = 255.0 * (grey - ct_min) / ct_width
    grey = wp.clamp(grey, 0.0, 255.0)

    if count_[0] > 1.0 and count_[1] > 1.0:
        r, g, b = 0.0, 255.0, 255.0
    elif count_[0] > 1.0:
        r, g, b = 0.0, 255.0, 0.0
    elif count_[1] > 1.0:
        r, g, b = 0.0, 0.0, 255.0
    elif count_[0] > 0.0:
        r, g, b = 127.0, 255.0, 127.0
    elif count_[1] > 0.0:
        r, g, b = 0.0, 127.0, 255.0
    else:
        r, g, b = grey, grey, grey

    image_2[i, j] = wp.vec3ub(wp.uint8(r), wp.uint8(g), wp.uint8(b))
