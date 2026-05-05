import argparse
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tomlkit
from tqdm import tqdm

from b0_config import client_pairs


def main(config_file: str, it: dict):
    # if 'hip_align' in it:
    #     return

    prl = it['prl']
    pid, rl = prl.split('_')

    cfg_path = Path(config_file)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    if it.get('excluded', False):
        return

    import numpy as np
    import warp as wp
    import trimesh
    import pyvista as pv
    from minio import Minio, S3Error
    from kernel import compute_sdf, icp
    from PIL import Image

    client = Minio(**cfg['minio']['client'])
    saved = it['align']

    metal_meshes = {'femur': [], 'hip': []}
    bone_meshes = {'femur': [], 'hip': []}

    with tempfile.TemporaryDirectory() as tdir:
        for op in ('pre', 'post'):
            for part in ('femur', 'hip'):
                if op == 'post':
                    try:
                        f = Path(tdir) / 'metal.stl'
                        object_name = '/'.join([pid, rl, op, part, f.name])
                        client.fget_object('pair', object_name, f.as_posix())

                        mesh = trimesh.load_mesh(f.as_posix())
                        metal_meshes[part].append(mesh)
                    except (S3Error, Exception):
                        if part in ('femur',):
                            raise RuntimeError(f'{op} {part} {f.name} 下载失败')

                        metal_meshes[part].append(None)
                else:
                    metal_meshes[part].append(None)

                try:
                    f = Path(tdir) / 'bone.stl'
                    object_name = '/'.join([pid, rl, op, part, f.name])
                    client.fget_object('pair', object_name, f.as_posix())
                except S3Error:
                    raise RuntimeError(f'{op} {part} {f.name} 下载失败')

                mesh = trimesh.load_mesh(f.as_posix())
                if not mesh.is_empty:
                    mesh = list(sorted(
                        mesh.split(only_watertight=False),
                        key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]), reverse=True,
                    ))[0]
                else:
                    raise RuntimeError(f'{op} {part} {f.name} 空网格体')
                bone_meshes[part].append(mesh)

    save = {'femur': {}, 'hip': {}}

    for part_id, part in enumerate(('hip', 'femur')):
        sizes = [np.array(it['roi'][part][_]['size']) for _ in ('pre', 'post')]
        spacings = [np.array(it['roi'][part][_]['spacing']) for _ in ('pre', 'post')]
        origins = [np.array(it['roi'][part][_]['origin']) for _ in ('pre', 'post')]

        post_mesh: trimesh.Trimesh = bone_meshes[part][1].copy()

        zl = [round(_.bounds[1][2] - _.bounds[0][2]) for _ in bone_meshes[part]]

        post_mesh_outlier: trimesh.Trimesh | None
        if part in ('femur',):
            d_proximal: int = saved.get(part, {}).get('d_proximal', min(zl[1], 15))

            # 根据 Z 轴高度裁剪术后网格，只保留用于配准的稳定骨干部分
            if d_proximal > 0 or zl[0] < zl[1]:
                z_max = post_mesh.bounds[1][2] - d_proximal
                z_min = z_max - zl[0]

                z = post_mesh.vertices[:, 2]
                mask = (z_min <= z) & (z <= z_max)
                post_mesh.update_faces(np.all(mask[post_mesh.faces], axis=1))
                post_mesh.remove_unreferenced_vertices()

                # 记录被裁掉的部分以便在 3D 中对比显示
                mask = ~mask
                post_mesh_outlier: trimesh.Trimesh = bone_meshes[part][1].copy()
                post_mesh_outlier.update_faces(np.all(mask[post_mesh_outlier.faces], axis=1))
                post_mesh_outlier.remove_unreferenced_vertices()

                if post_mesh.is_empty:
                    raise RuntimeError('近端裁剪过多')

            else:
                post_mesh_outlier = None
        else:
            d_proximal = 0
            post_mesh_outlier = None

        # 采样点纵向范围
        zl.append(round(post_mesh.bounds[1][2] - post_mesh.bounds[0][2]))
        _min, _max = int(d_proximal), int(d_proximal) + int(min(zl[0], zl[2]))

        _def = saved.get(part, {}).get('d_sample_range', (_min, _max))
        _def = (max(_min, min(int(_def[0]), _max)), max(_min, min(int(_def[1]), _max)))

        if _def[0] > _def[1]: _def = (_min, _max)

        d_sample_range = _def

        # 避开金属假体的距离
        d_metal: int = saved.get(part, {}).get('d_metal', 5 if part in ('femur',) else 15)

        max_dist = float(np.linalg.norm(sizes[1] * spacings[1]))
        d = wp.full((len(post_mesh.vertices),), max_dist, float)

        if metal_meshes[part][1] is not None:
            # 使用 Warp 计算术后骨骼网格顶点到金属假体网格的距离 (SDF)
            metal = wp.Mesh(wp.array(metal_meshes[part][1].vertices, wp.vec3),
                            wp.array(metal_meshes[part][1].faces.flatten(), wp.int32))

            wp.launch(compute_sdf, d.shape, [
                wp.uint64(metal.id), wp.array(post_mesh.vertices, wp.vec3), d, max_dist,
            ])

        d = d.numpy()

        if part == 'femur':
            z0 = post_mesh.bounds[1][2] - post_mesh.vertices[:, 2]
            z_mask = (d_sample_range[0] - d_proximal <= z0) & (z0 <= d_sample_range[1] - d_proximal)
        else:
            z0 = post_mesh.vertices[:, 2] - post_mesh.bounds[0][2]
            z_mask = (d_sample_range[0] <= z0) & (z0 <= d_sample_range[1])

        # 权重计算：距离金属越远、在采样范围内的顶点权重越高
        _ = d - d_metal
        _ = np.clip(_, 0, max(d_metal, 1e-6))
        _ *= z_mask

        if (n := min(int(np.sum(_ > 0)), 10000)) < 100:
            raise RuntimeError(f'采样点过少 {n}')

        # 随机采样 10000 个顶点用于 ICP 配准
        _ = _ / _.sum()
        _ = np.random.choice(len(post_mesh.vertices), size=n, replace=False, p=_)
        vertices = post_mesh.vertices[_]

        # 初始对齐：将术后网格的顶部对齐到术前网格的顶部（Z 轴平移）
        matrix = np.identity(4)
        if part == 'femur':
            matrix[2, 3] = bone_meshes[part][0].bounds[1][2] - post_mesh.bounds[1][2]
        else:
            matrix[2, 3] = bone_meshes[part][0].bounds[0][2] - post_mesh.bounds[0][2]

        # 执行 ICP (Iterative Closest Point) 算法精细对齐
        matrix, _, mse, iters = icp(
            vertices, bone_meshes[part][0], matrix, 1e-5, 2000,
            **dict(reflection=False, scale=False),
        )

        # 核心逻辑：计算从术后“原始图像坐标系”到术前“原始图像坐标系”的全局变换矩阵 (g_matrix)
        # ICP 得到的是 ROI 局部坐标系下的变换，需要结合 ROI 在原图中的偏移量进行还原
        offset = [origins[_] for _ in range(2)]

        pre = np.identity(4)
        pre[:3, 3] = offset[0]

        post_inv = np.identity(4)
        post_inv[:3, 3] = -offset[1]

        g_matrix = pre @ matrix @ post_inv

        xform = np.array(wp.transform_from_matrix(wp.mat44(g_matrix)), dtype=float).tolist()

        save[f'{part}_align'] = xform

        if part in ('femur',):
            save[part]['d_proximal'] = d_proximal

        save[part]['d_sample_range'] = d_sample_range
        save[part]['d_metal'] = d_metal
        save[part]['post_points'] = len(vertices)
        save[part]['iterations'] = int(iters)
        save[part]['mse'] = float(mse)

        # 使用 PyVista 构建三维场景
        b = np.array(post_mesh.bounds)
        b[1][2] += d_proximal

        x, y, z = [b[1][_] - b[0][_] for _ in (0, 1, 2)]
        wx, wy, h = [round(_ * 5) for _ in (x, y, z)]

        pl = pv.Plotter(
            off_screen=True, border=False, window_size=[768, 768],
            line_smoothing=True, point_smoothing=True, polygon_smoothing=True,
        )
        pl.enable_parallel_projection()  # 使用正交投影便于观察几何关系
        pl.enable_depth_peeling()
        pl.enable_anti_aliasing('msaa')

        # 添加各种组件到场景
        if post_mesh_outlier is not None and len(post_mesh_outlier.faces):
            pl.add_mesh(post_mesh_outlier, color='green')  # noqa 术后被裁剪掉的部分（深绿）
        pl.add_mesh(post_mesh, color='lightgreen')  # noqa 术后用于配准的部分（浅绿）

        pre_mesh: trimesh.Trimesh = bone_meshes[part][0].copy()
        pre_mesh.apply_transform(np.linalg.inv(matrix))  # 将术前网格逆变换到术后坐标系对比
        pl.add_mesh(pre_mesh, color='lightyellow')  # noqa 术前参考网格（浅黄）
        pl.add_points(vertices, color='crimson', render_points_as_spheres=True, point_size=3)  # 实际采样点（深红）

        pl.camera_position = 'xz'
        pl.reset_camera(bounds=b.T.flatten())
        pl.camera.parallel_scale = (b[1][2] - b[0][2]) * 0.6
        pl.reset_camera_clipping_range()
        pl.render()

        # 渲染并拼接正侧位（AP & Lateral）视图
        sil = None
        if metal_meshes[part][1] is not None:
            metal_actor = pl.add_mesh(metal_meshes[part][1], color='lightblue')  # 金属假体（浅蓝）
            sil = pl.add_silhouette(metal_actor.GetMapper().GetInput(), color='lightgray')  # noqa

        imgs = []
        # 循环生成两个视角的截图：正面 (0度) 和侧面 (90或-90度)
        for i, deg in enumerate([0, 90 if rl == 'R' else -90]):
            for actor in pl.actors.values():
                actor.SetVisibility(False)
            if sil is not None:
                sil.SetVisibility(True)

            pl.window_size = [[wx, wy][i], h]

            pl.camera_position = 'xz'
            pl.reset_camera(bounds=b.T.flatten())
            pl.camera.Azimuth(deg)
            pl.camera.parallel_scale = (b[1][2] - b[0][2]) * 0.6
            pl.reset_camera_clipping_range()
            pl.render()
            a = np.array(pl.screenshot(return_img=True)).copy()

            [pl.actors[_].SetVisibility(True) for _ in pl.actors].clear()

            pl.reset_camera_clipping_range()
            pl.render()
            c = np.array(pl.screenshot(return_img=True)).copy()

            # 将金属假体的剪影合并到截图上，确保假体始终可见
            mask = (a != pl.background_color.int_rgb).any(axis=-1)
            c[mask] = a[mask]
            imgs.append(c)

        # 横向拼接图像并显示在 Streamlit
        stack = np.hstack(imgs)

        save_file = Path(cfg['dataset']['root']) / 'align_pred' / part / f'{prl}.png'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(stack).save(save_file)

        pl.close()

    # 提交
    for k, v in save.items():
        if isinstance(v, dict) and k in saved and isinstance(saved[k], dict):
            saved[k].update(v)
        else:
            saved[k] = v
    data = tomlkit.dumps(saved).encode('utf-8')
    from io import BytesIO
    client.put_object('pair', '/'.join([pid, rl, 'align.toml']), BytesIO(data), len(data))


def launch(cfg_path: str, max_workers: int):
    client, pairs = client_pairs(cfg_path, ['align'])

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(main, cfg_path, it): prl for prl, it in pairs.items()}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    fu.result()
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=1)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()

    launch(args.config, args.max_workers)
