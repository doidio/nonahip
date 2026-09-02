"""Preoperative CT → TotalSegmentator → side ROI → 1 mm training-space volume."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import itk
import numpy as np
import warp as wp

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import define  # noqa: E402
from kernel import diff_dmc, resample_pre  # noqa: E402

CORE_SIZE = 32
SIDE_LABELS = {
    'L': {'femur': define.ct_seg_femur_left, 'hip': define.ct_seg_hip_left},
    'R': {'femur': define.ct_seg_femur_right, 'hip': define.ct_seg_hip_right},
}
SIDE_NAMES = {'L': '左侧', 'R': '右侧'}


def run_totalsegmentator(image_path: str | Path, label_path: str | Path):
    image_path = Path(image_path)
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        'from kernel import itk_monkey_patch\n'
        'itk_monkey_patch()\n'
        'from totalsegmentator.python_api import totalsegmentator\n'
        f'totalsegmentator({image_path.as_posix()!r}, {label_path.as_posix()!r}, True, task="total", quiet=True)\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=str(_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or '').strip()
        raise RuntimeError(detail or 'TotalSegmentator 失败')
    if not label_path.exists() or label_path.stat().st_size == 0:
        raise RuntimeError('TotalSegmentator 未写出分割结果')
    return label_path


def load_itk_xyz(path: str | Path, pixel_type=None):
    path = Path(path)
    image = itk.imread(path.as_posix()) if pixel_type is None else itk.imread(path.as_posix(), pixel_type)
    origin = np.array(itk.origin(image), dtype=np.float64)
    spacing = np.array(itk.spacing(image), dtype=np.float64)
    array = itk.array_from_image(image).transpose(2, 1, 0)
    return array, origin, spacing


def side_voxel_counts(label_xyz: np.ndarray):
    counts = {}
    for side, parts in SIDE_LABELS.items():
        femur_n = int(np.sum(label_xyz == parts['femur']))
        hip_n = int(np.sum(label_xyz == parts['hip']))
        counts[side] = {
            'femur': femur_n,
            'hip': hip_n,
            'ok': femur_n > 0 and hip_n > 0,
        }
    return counts


def available_sides(label_xyz: np.ndarray):
    counts = side_voxel_counts(label_xyz)
    return [side for side in ('L', 'R') if counts[side]['ok']]


def bone_normalize_np(ct_value):
    ct = np.asarray(ct_value, dtype=np.float32)
    out = np.full(ct.shape, -2.0, dtype=np.float32)
    mask = (ct >= -1000.0) & (ct < 0.0)
    out[mask] = -2.0 + (ct[mask] - 0.0) / 1000.0 * 0.5
    mask = (ct >= 0.0) & (ct < 150.0)
    out[mask] = -1.5 + (ct[mask] - 0.0) / 150.0 * 0.5
    mask = (ct >= 150.0) & (ct < 650.0)
    out[mask] = -1.0 + (ct[mask] - 150.0) / 500.0 * 1.0
    mask = (ct >= 650.0) & (ct < 1150.0)
    out[mask] = 0.0 + (ct[mask] - 650.0) / 500.0 * 0.5
    mask = (ct >= 1150.0) & (ct < 3150.0)
    out[mask] = 0.5 + (ct[mask] - 1150.0) / 2000.0 * 0.5
    out[ct >= 3150.0] = 1.0
    return out


def _largest_label_box(mask_xyz):
    from scipy import ndimage

    labeled, n = ndimage.label(mask_xyz)
    if n == 0:
        ijk = np.argwhere(mask_xyz)
    else:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        ijk = np.argwhere(labeled == int(np.argmax(counts)))
    return np.array([ijk.min(axis=0), ijk.max(axis=0) + 1], dtype=int)


def _extract_part_roi(image_xyz, label_xyz, origin, spacing, label):
    if np.sum(label_xyz == label) == 0:
        raise RuntimeError(f'自动分割不包含标签 {label}')

    ijk = np.argwhere(label_xyz == label)
    box = np.array([ijk.min(axis=0), ijk.max(axis=0) + 1], dtype=int)

    for count in range(2):
        box[0] = np.maximum(box[0], 0)
        box[1] = np.minimum(box[1], image_xyz.shape)

        roi_image = image_xyz[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]].copy()
        roi_label = label_xyz[box[0, 0] : box[1, 0], box[0, 1] : box[1, 1], box[0, 2] : box[1, 2]]
        roi_image[(roi_label != label) & (roi_image > define.ct_bone_best)] = define.ct_min

        try:
            bone_mesh = diff_dmc(wp.from_numpy(roi_image, wp.float32), np.zeros(3), spacing, define.ct_bone_best)
            if bone_mesh.is_empty:
                raise RuntimeError('empty mesh')
            bone_mesh = list(
                sorted(
                    bone_mesh.split(only_watertight=False),
                    key=lambda mesh: np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]),
                    reverse=True,
                )
            )[0]
            if count > 0:
                break
            box = box[0] + np.array(
                [np.floor(bone_mesh.bounds[0] / spacing), np.ceil(bone_mesh.bounds[1] / spacing)]
            ).astype(int)
        except Exception:
            local = _largest_label_box(roi_label == label)
            box = np.array([box[0] + local[0], box[0] + local[1]])
            break

    roi_origin = origin + spacing * box[0]
    roi_size = box[1] - box[0]
    return {
        'origin': roi_origin.astype(float),
        'spacing': np.asarray(spacing, dtype=float),
        'size': roi_size.astype(int),
        'box': box,
    }


def extract_side_rois(image_xyz, label_xyz, origin, spacing, side: str):
    parts = SIDE_LABELS[side]
    return {
        name: _extract_part_roi(image_xyz, label_xyz, origin, spacing, label)
        for name, label in parts.items()
    }


def _physical_box(roi):
    origin = np.asarray(roi['origin'], dtype=np.float64)
    spacing = np.asarray(roi['spacing'], dtype=np.float64)
    size = np.asarray(roi['size'], dtype=np.float64)
    return np.array([origin, origin + spacing * size])


def combine_pre_roi_grid(rois: dict):
    boxes = np.array([_physical_box(rois['hip']), _physical_box(rois['femur'])])
    roi_box = np.array([boxes[:, 0].min(0), boxes[:, 1].max(0)])
    padding = define.roi_spacing + define.sdf_t
    roi_box[0] -= padding
    roi_box[1] += padding

    extents = roi_box[1] - roi_box[0]
    roi_size = np.ceil(extents / define.roi_spacing).astype(int) + np.array([2, 2, 0])
    roi_size = np.ceil(roi_size / CORE_SIZE).astype(int) * CORE_SIZE
    roi_size = np.maximum(roi_size, CORE_SIZE)
    roi_origin = (roi_box[0] + roi_box[1]) * 0.5 - 0.5 * define.roi_spacing * roi_size
    return roi_origin.astype(np.float64), np.array([define.roi_spacing] * 3, dtype=np.float64), roi_size.astype(int)


def _cuda_ok():
    try:
        return int(wp.get_cuda_device_count()) > 0
    except Exception:
        return False


def _resample_pre_numpy(image_xyz, origin, spacing, roi_origin, roi_spacing, roi_size):
    from scipy.ndimage import map_coordinates

    roi_size = np.asarray(roi_size, dtype=int)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    roi_origin = np.asarray(roi_origin, dtype=np.float64)
    roi_spacing = np.asarray(roi_spacing, dtype=np.float64)
    axes = [roi_origin[i] + np.arange(roi_size[i]) * roi_spacing[i] for i in range(3)]
    grid = np.meshgrid(*axes, indexing='ij')
    coords = np.stack([(grid[i] - origin[i]) / spacing[i] for i in range(3)])
    sampled = map_coordinates(
        np.ascontiguousarray(image_xyz.astype(np.float32)),
        coords,
        order=1,
        mode='constant',
        cval=float(define.ct_min),
    )
    return bone_normalize_np(sampled)


def resample_pre_volume(image_xyz, origin, spacing, roi_origin, roi_spacing, roi_size):
    if _cuda_ok():
        volume = wp.Volume.load_from_numpy(
            np.ascontiguousarray(image_xyz.astype(np.float32)),
            bg_value=float(define.ct_min),
        )
        roi_image = wp.full(tuple(int(x) for x in roi_size), -1.0, dtype=wp.float32)
        wp.launch(
            resample_pre,
            tuple(int(x) for x in roi_size),
            [
                roi_image,
                wp.vec3(*np.asarray(roi_origin, dtype=np.float32)),
                wp.vec3(*np.asarray(roi_spacing, dtype=np.float32)),
                volume.id,
                wp.vec3(*np.asarray(origin, dtype=np.float32)),
                wp.vec3(*np.asarray(spacing, dtype=np.float32)),
            ],
        )
        return roi_image.numpy()
    return _resample_pre_numpy(image_xyz, origin, spacing, roi_origin, roi_spacing, roi_size)


def save_pre_nifti(path: str | Path, image_xyz: np.ndarray, origin, spacing):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = itk.image_from_array(np.ascontiguousarray(image_xyz.transpose(2, 1, 0)))
    image.SetOrigin(list(np.asarray(origin, dtype=float)))
    image.SetSpacing(list(np.asarray(spacing, dtype=float)))
    itk.imwrite(image, path.as_posix())
    return path


def prepare_preoperative_volume(image_xyz, label_xyz, origin, spacing, side: str, out_path: str | Path, progress=None):
    def _progress(value, text):
        if progress is not None:
            progress(value, text)

    _progress(0.1, '定位术区')
    rois = extract_side_rois(image_xyz, label_xyz, origin, spacing, side)
    roi_origin, roi_spacing, roi_size = combine_pre_roi_grid(rois)
    _progress(0.45, '对齐影像')
    pre_xyz = resample_pre_volume(image_xyz, origin, spacing, roi_origin, roi_spacing, roi_size)
    _progress(0.85, '保存术区')
    save_pre_nifti(out_path, pre_xyz, roi_origin, roi_spacing)
    return {
        'side': side,
        'rois': {
            name: {
                'origin': roi['origin'].tolist(),
                'spacing': roi['spacing'].tolist(),
                'size': roi['size'].tolist(),
            }
            for name, roi in rois.items()
        },
        'origin': roi_origin.tolist(),
        'spacing': roi_spacing.tolist(),
        'size': roi_size.tolist(),
        'path': Path(out_path),
        'image': pre_xyz,
    }

