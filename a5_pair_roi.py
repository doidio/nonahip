import argparse
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tomlkit
from b0_config import load_pairs
from tqdm import tqdm


def main(cfg: dict, prl: str, it: dict):
    import itk
    import numpy as np
    import warp as wp
    from define import (
        ct_bone_best,
        ct_metal,
        ct_min,
        ct_seg_femur_left,
        ct_seg_femur_right,
        ct_seg_hip_left,
        ct_seg_hip_right,
    )
    from kernel import diff_dmc

    pid, rl = prl.split('_')
    root = Path(cfg['dataset']['root'])

    fem_label = {'R': ct_seg_femur_right, 'L': ct_seg_femur_left}[rl]
    hip_label = {'R': ct_seg_hip_right, 'L': ct_seg_hip_left}[rl]

    roi_data = {}
    for op in ('pre', 'post'):
        roi_data[op] = {}

        # 读取分割图像
        f = root / 'total' / pid / it['pair'][op]
        total = itk.imread(f.as_posix(), itk.UC)
        total = itk.array_from_image(total).transpose(2, 1, 0)

        # 读取原始 CT 图像
        f = root / 'nii' / pid / it['pair'][op]
        image = itk.imread(f.as_posix(), itk.SS)

        # size = np.array(itk.size(image), float)
        spacing = np.array(itk.spacing(image), float)
        origin = np.array(itk.origin(image), float)

        image = itk.array_from_image(image).transpose(2, 1, 0)

        for part, label in (('femur', fem_label), ('hip', hip_label)):
            roi_data[op][part] = {}

            if np.sum(total_roi := (total == label)) == 0:
                raise RuntimeError(f'{op} 自动分割不包含 {part} {label}')

            ijk = np.argwhere(total_roi)
            box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1])

            for count in range(2):
                box[0] = np.maximum(box[0], 0)
                box[1] = np.minimum(box[1], image.shape)

                # 提取子区域（可能有离群噪点）
                roi_image = image[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]].copy()
                roi_total = total[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]].copy()

                # 非目标区域的高亮部分（如邻近骨骼）置为背景，避免干扰配准
                roi_image[np.where((roi_total != label) & (roi_image > ct_bone_best))] = ct_min

                # 提取骨骼网格
                bone_mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_bone_best)

                # 对角线最大连通体（排除离群噪点）
                if bone_mesh.is_empty:
                    raise RuntimeError(f'{op} {part} 不包含骨骼')

                bone_mesh = list(
                    sorted(
                        bone_mesh.split(only_watertight=False),
                        key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]),
                        reverse=True,
                    )
                )[0]

                if count > 0:
                    break

                # 修正子区域
                box = box[0] + np.array(
                    [np.floor(bone_mesh.bounds[0] / spacing), np.ceil(bone_mesh.bounds[1] / spacing)]
                ).astype(int)

            roi_origin = origin + spacing * box[0]
            roi_spacing = spacing.copy()
            roi_size = box[1] - box[0]

            # 如果是术后数据，提取金属假体网格
            if op == 'post':
                # 使用 GPU 加速的 Marching Cubes 提取金属等值面
                metal_mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, ct_metal)  # type: ignore
                if metal_mesh.is_empty and part in ('femur',):
                    raise RuntimeError(f'{op} {part} 不包含金属')
            else:
                metal_mesh = None

            # 存储
            roi_image = itk.image_from_array(np.ascontiguousarray(roi_image.transpose(2, 1, 0)))  # type: ignore
            roi_image.SetOrigin(roi_origin)
            roi_image.SetSpacing(roi_spacing)

            rm = root / 'pair' / pid / rl / part
            if rm.exists():
                shutil.rmtree(rm)

            rm = root / 'pair' / pid / rl / op / part / 'roi.toml'
            if rm.exists():
                rm.unlink()

            f = root / 'pair' / pid / rl / op / part / 'roi.nii.gz'
            f.parent.mkdir(parents=True, exist_ok=True)
            itk.imwrite(roi_image, f.as_posix())

            if metal_mesh and not metal_mesh.is_empty:
                f = f.parent / 'metal.stl'
                metal_mesh.export(f.as_posix())

            f = f.parent / 'bone.stl'
            bone_mesh.export(f.as_posix())

            roi_data[op][part] = {
                'origin': roi_origin.tolist(),
                'spacing': roi_spacing.tolist(),
                'size': roi_size.tolist(),
            }

    f = root / 'pair' / pid / rl / 'roi.toml'
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(tomlkit.dumps(roi_data), 'utf-8')


def launch(config_file: str, max_workers: int):
    cfg = Path(config_file)
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()
    cfg, pairs = load_pairs(cfg, [])

    # for prl in ('1446661_L',):
    #     main(root, prl, pairs[prl])
    # return

    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
        futures = {executor.submit(main, cfg, prl, pairs[prl]): prl for prl in pairs}

        try:
            for fu in tqdm(as_completed(futures), total=len(futures)):
                try:
                    fu.result()
                except Exception as _:
                    warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

        except KeyboardInterrupt:
            print('Keyboard interrupted terminating...')
            executor.shutdown(wait=False)
            for future in futures:
                future.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    launch(args.config, args.max_workers)
