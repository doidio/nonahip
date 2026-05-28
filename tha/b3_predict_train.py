import argparse
import multiprocessing
import time
from pathlib import Path

import itk
import numpy as np
import tomlkit
import trimesh
import warp as wp
from b0_config import load_pairs
from define import ct_metal, ct_min, roi_spacing, sdf_t
from kernel import diff_dmc, fast_drr, resample_metal, resample_roi
from PIL import Image
from tqdm import tqdm

core_size = 32


def preload(cfg: dict, it: dict):
    prl = it['prl']
    pid, rl = prl.split('_')

    dataset = Path(cfg['dataset']['root'])
    train = Path(cfg['train']['root'])
    context, align = it['context'], it['align']

    # 读取数据
    ct_images, sizes, spacings, origins = [], [], [], []

    for op in ('pre', 'post'):
        f = dataset / 'nii' / pid / it['pair'][op]
        if not f.exists():
            raise RuntimeError(f'Missing {op} data: {f}')

        image = itk.imread(f.as_posix(), itk.SS)

        size = np.array(itk.size(image), float)
        spacing = np.array(itk.spacing(image), float)
        origin = np.array(itk.origin(image), float)

        sizes.append(size)
        spacings.append(spacing)
        origins.append(origin)

        image = itk.array_from_image(image).transpose(2, 1, 0)
        ct_images.append(image)

    volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=ct_min) for _ in range(2)]

    # 配准变换
    xforms = [wp.transform(*align[f'{part}_align']) for part in ('hip', 'femur')]
    xforms_inv = [wp.transform_inverse(_) for _ in xforms]  # type: ignore

    cup_radius = int(context['cup_outer_best'] if 'cup_outer_best' in context else context['cup_outer']) * 0.5
    cup_center = np.array(context['cup_center'], float)
    cup_center = [np.array(wp.transform_point(xforms[_], wp.vec3(cup_center))) for _ in range(2)]  # type: ignore

    cup_axis = np.array(context['cup_axis'], float)
    cup_axis = [np.array(wp.transform_vector(xforms[_], wp.vec3(cup_axis))) for _ in range(2)]  # type: ignore

    head_radius = int(context['head_outer']) * 0.5
    head_center = np.array(context['head_center'], float)
    head_center = [np.array(wp.transform_point(xforms[_], wp.vec3(head_center))) for _ in range(2)]  # type: ignore

    # 采样区域
    roi_boxes = []
    for i, part in enumerate(('hip', 'femur')):
        cup_box = [cup_center[i] - cup_radius, cup_center[i] + cup_radius]

        origin_pre = np.array(it['roi']['pre'][part]['origin'])
        spacing_pre = np.array(it['roi']['pre'][part]['spacing'])
        size_pre = np.array(it['roi']['pre'][part]['size'])
        box_pre = np.array([origin_pre, origin_pre + spacing_pre * size_pre])

        origin_post = np.array(it['roi']['post'][part]['origin'])
        spacing_post = np.array(it['roi']['post'][part]['spacing'])
        size_post = np.array(it['roi']['post'][part]['size'])

        # 计算术后包围盒在其局部坐标系下的中心和半边长
        center_post_local = origin_post + spacing_post * size_post * 0.5
        extents_post_local = spacing_post * size_post * 0.5

        # 变换中心点到术前(世界)坐标系
        center_post_world = np.array(wp.transform_point(xforms[i], wp.vec3(*center_post_local)))  # type: ignore

        # 提取旋转矩阵
        axes_world = [
            np.array(wp.transform_vector(xforms[i], wp.vec3(*axis)))  # type: ignore
            for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ]
        R = np.column_stack(axes_world)

        # 混合策略：X-Y 方向外接以避免裁剪，Z 方向使用内接以避开斜面裁剪
        h_circum = np.abs(R) @ extents_post_local
        h_inscribe = extents_post_local / np.maximum(1.0, np.sum(np.abs(R), axis=0))

        h_final = np.array([h_circum[0], h_circum[1], h_inscribe[2]])
        box_post = np.array([center_post_world - h_final, center_post_world + h_final])

        # 保留交集，确保不超出术前图像的扫描范围
        box_common = np.array([np.maximum(box_pre[0], box_post[0]), np.minimum(box_pre[1], box_post[1])])

        padding = roi_spacing + sdf_t
        box = np.array([box_common, [cup_box[0] - padding, cup_box[1] + padding]])
        box = np.array([box[:, 0].min(0), box[:, 1].max(0)])

        roi_boxes.append(box)

    roi_boxes = np.array(roi_boxes)
    roi_box = np.array([roi_boxes[:, 0].min(0), roi_boxes[:, 1].max(0)])

    extents = roi_box[1] - roi_box[0]

    roi_size = np.ceil(extents / roi_spacing).astype(int) + [2, 2, 0]
    roi_size = np.ceil(roi_size / core_size).astype(int) * core_size
    max_dist = wp.float32(np.linalg.norm(roi_size * roi_spacing))  # type: ignore
    roi_origin = (roi_box[0] + roi_box[1]) * 0.5 - 0.5 * roi_spacing * roi_size

    roi_images = wp.full((*roi_size,), wp.vec3(-1.0), dtype=wp.vec3)
    hip_metal = wp.full((*roi_size,), -1.0, wp.float32)
    femur_metal = wp.full((*roi_size,), -1.0, wp.float32)

    wp.launch(
        resample_roi,
        (*roi_size,),
        [
            roi_images,
            roi_origin,
            wp.vec3(roi_spacing),
            roi_boxes[0][0],
            roi_boxes[0][1],
            roi_boxes[1][0],
            roi_boxes[1][1],
            xforms_inv[0],
            xforms_inv[1],
            volumes[0].id,
            origins[0],
            spacings[0],
            volumes[1].id,
            origins[1],
            spacings[1],
            hip_metal,
            femur_metal,
            ct_metal,
        ],
    )

    roi_images = roi_images.numpy()
    pre_image, post_images = roi_images[..., 0], [roi_images[..., 1], roi_images[..., 2]]

    # 重建假体
    meshes = []
    for i, part in enumerate(('cup', 'stem')):
        mesh = diff_dmc([hip_metal, femur_metal][i], roi_origin, roi_spacing, 0.0)

        if not mesh.is_empty:
            ls = [
                m
                for m in sorted(
                    mesh.split(only_watertight=False),
                    key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]),
                    reverse=True,
                )
                if np.linalg.norm(m.vertices - head_center[i], axis=1).min() <= cup_radius
            ]

            if len(ls):
                mesh: trimesh.Trimesh = trimesh.util.concatenate(ls)  # noqa
                mesh.fix_normals()
            else:
                raise RuntimeError(f'{part} metal is far')
        else:
            raise RuntimeError(f'{part} metal is empty')

        wp_mesh = wp.Mesh(wp.array(mesh.vertices, dtype=wp.vec3), wp.array(mesh.faces.flatten(), dtype=wp.int32), support_winding_number=True)
        meshes.append(wp_mesh)

    # 采样假体
    hip_image = wp.full((*roi_size,), -1.0, wp.float32)
    femur_image = wp.full((*roi_size,), -1.0, wp.float32)

    wp.launch(
        resample_metal,
        (*roi_size,),
        [
            hip_image,
            femur_image,
            roi_origin,
            wp.vec3(roi_spacing),
            meshes[0].id,
            meshes[1].id,
            cup_center[0],
            cup_center[1],
            cup_axis[0],
            cup_axis[1],
            head_center[0],
            head_center[1],
            cup_radius,
            head_radius,
            sdf_t,
            max_dist,
        ],
    )

    for f, image in (
        (train / 'dataset' / 'metal' / f'{prl}_cup.stl', hip_image),
        (train / 'dataset' / 'metal' / f'{prl}_stem.stl', femur_image),
    ):
        f.parent.mkdir(parents=True, exist_ok=True)

        mesh = diff_dmc(image, roi_origin, roi_spacing, 0.0)
        mesh.export(f)

    hip_image = hip_image.numpy()
    femur_image = femur_image.numpy()

    # 存档
    for f, image in (
        (train / 'dataset' / 'pre' / f'{prl}.nii.gz', pre_image),
        (train / 'dataset' / 'post_align_hip' / f'{prl}.nii.gz', post_images[0]),
        (train / 'dataset' / 'post_align_femur' / f'{prl}.nii.gz', post_images[1]),
        (train / 'dataset' / 'metal' / f'{prl}_cup.nii.gz', hip_image),
        (train / 'dataset' / 'metal' / f'{prl}_stem.nii.gz', femur_image),
    ):
        f.parent.mkdir(parents=True, exist_ok=True)

        if image.ndim == 3:
            image_itk = itk.image_from_array(np.ascontiguousarray(image.transpose(2, 1, 0)))
        else:
            image_itk = itk.image_from_array(np.ascontiguousarray(image.transpose(2, 1, 0, 3)), is_vector=True)

        image_itk.SetOrigin(roi_origin)
        image_itk.SetSpacing(roi_spacing)
        itk.imwrite(image_itk, f.as_posix())

    # 快照
    stack = []
    axis = 1  # 冠状面投影
    for image in (pre_image, *post_images, hip_image, femur_image):
        img = fast_drr(image + 1.0, axis, th=(0.0, 2.0), mode='mean')
        # 转置并翻转，确保解剖方位正确（Superior 在上）
        img = np.flipud(img.transpose(1, 0, 2))
        stack.append(img)

    f = train / 'dataset' / 'png' / f'{prl}.png'
    f.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.hstack(stack)).save(f)

    return roi_origin, roi_spacing, [pre_image, *post_images, hip_image, femur_image]


def submit(cfg: dict, it: dict, *args):
    prl = it['prl']
    pid, rl = prl.split('_')


def main(cfg: dict, it: dict):
    if it.get('excluded', False):
        return

    submit(cfg, it, *preload(cfg, it))

    import gc

    gc.collect()
    time.sleep(0.5)  # 短暂休眠以确保资源释放完毕


def launch(config_file: str, max_workers: int):
    cfg = tomlkit.loads(Path(config_file).read_text('utf-8')).unwrap()

    cfg, pairs = load_pairs(cfg, ['roi', 'context', 'align'])

    # for _ in tqdm(['3212905_L', '1004333_L', '1105282_R', '1203526_R']):
    #     main(cfg, pairs[_])
    # return

    # 按股骨柄型号抽取验证集和测试集
    stem = {}
    for prl, it in pairs.items():
        if it.get('excluded', False):
            continue

        if 'femoral_spec' not in it['context']:
            continue

        spec = it['context']['femoral_spec'][0]
        if spec not in stem:
            stem[spec] = []
        stem[spec].append(prl)

    trains, vals, tests = {}, {}, {}
    for spec, ls in stem.items():
        for _ in range(10):
            if len(ls):
                trains[ls.pop()] = spec

            if len(ls):
                vals[ls.pop()] = spec

            if len(ls) and _ < 3:
                tests[ls.pop()] = spec

    cfg['val'] = vals
    cfg['test'] = tests
    Path(config_file).write_text(tomlkit.dumps(cfg), 'utf-8')

    tasks = [(cfg, it) for prl, it in pairs.items()]

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        for _ in tqdm(pool.imap_unordered(_process, tasks), total=len(tasks)):
            pass


def _process(args):
    """
    多进程任务的包装函数，用于捕获并处理键盘中断异常。
    """
    try:
        main(*args)
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--max_workers', type=int, default=24)
    args = parser.parse_args()

    launch(args.config, args.max_workers)
