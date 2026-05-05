import argparse
import multiprocessing
import tempfile
import time
from pathlib import Path

import itk
import numpy as np
import tomlkit
import trimesh
import warp as wp
from PIL import Image
from minio import Minio, S3Error
from tqdm import tqdm

from define import ct_metal, roi_spacing, sdf_t
from kernel import diff_dmc, resample_roi, fast_drr


def main(config: str, prl: str, pair: dict):
    """
    处理单个病例的术前/术后配对数据，提取感兴趣区域(ROI)并进行重采样对齐。
    
    :param config: 配置文件路径
    :param prl: 病例标识符 (例如: patientID_R 或 patientID_L)
    :param pair: 包含术前、术后数据路径及配准变换矩阵的字典
    """
    # 加载 TOML 配置文件
    cfg_path = Path(config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    # 初始化 Minio 客户端，用于从对象存储中读取数据
    client = Minio(**cfg['minio']['client'])

    # 设置数据集输出根目录
    dataset_root = Path(str(cfg['dataset']['root']))

    # 检查是否已经处理过（通过快照文件是否存在来判断），避免重复处理
    snapshot_file = dataset_root / 'snapshot' / f'{prl}.png'
    if snapshot_file.exists():
        return

    # 根据 pair 字典中的标记决定输出到验证集(val)还是训练集(train)目录
    subdir = 'val' if pair.get('is_val', False) else 'train'

    rl = prl.split('_')[1]

    # 初始化存储术前/术后图像属性的列表
    ct_images, roi_bounds, ct_femurs, sizes, spacings, origins = [], [], [], [], [], []

    # 遍历术前 (pre) 和术后 (post) 数据
    with tempfile.TemporaryDirectory() as tdir:
        for op, object_name in enumerate((pair['pre'], pair['post'])):
            f = Path(tdir) / 'image.nii.gz'
            try:
                client.fget_object('nii', object_name, f.as_posix())
            except S3Error:
                raise RuntimeError('Missing {} data'.format(['pre', 'post'][op]))

            image = itk.imread(f.as_posix(), itk.SS)

            size = np.array(itk.size(image), float)
            spacing = np.array(itk.spacing(image), float)
            origin = np.array(itk.origin(image), float)

            sizes.append(size)
            spacings.append(spacing)
            origins.append(origin)

            image = itk.array_from_image(image).transpose(2, 1, 0)
            ct_images.append(image)

    # 提取金属网格体
    _ = ct_femurs[1] & (ct_images[1] >= ct_metal)
    mesh = diff_dmc(wp.from_numpy(_, wp.float32), origins[1], spacings[1], 0.5)

    # 金属可能分离成髋臼杯、球头、股骨柄、膝关节假体，选范围最大的股骨柄以上
    if not mesh.is_empty:
        # 将网格拆分为独立的连通分量，并按包围盒对角线长度降序排序，取最大的组件（通常是股骨柄）
        ls = list(sorted(
            mesh.split(only_watertight=True),
            key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]), reverse=True,
        ))
        mesh: trimesh.Trimesh = ls[0]
        mesh.fix_normals()
    else:
        raise RuntimeError('No metal implant found in the post-op image.')

    # 解析术后到术前的配准变换矩阵
    if 'post_xform' in pair:
        post_xform = wp.transform(*pair['post_xform'])
        post_xform = np.array(wp.transform_to_matrix(post_xform), float).reshape((4, 4))
        post_xform = wp.transform_from_matrix(wp.mat44(post_xform))
    else:
        raise RuntimeError('Missing post_xform matrix in pair data.')

    # 计算目标 ROI 的轴向包围盒 (AABB)
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    extents += sdf_t * 2.0

    # 根据物理范围计算体素尺寸
    roi_size = np.ceil(extents / roi_spacing).astype(int)
    # 向上取整到 64 的倍数，以适配后续深度学习网络（如 U-Net/VAE）的多次下采样要求
    roi_size = np.ceil(roi_size / 64.0).astype(int) * 64

    # 计算 ROI 的起始物理坐标 (原点)
    origin = -0.5 * roi_spacing * roi_size

    # 构建 ROI 的变换矩阵（仅包含平移到中心点）
    roi_xform = np.identity(4)
    roi_xform[:3, 3] = center

    roi_xform = wp.transform_from_matrix(wp.mat44(roi_xform))

    # 将 numpy 数组加载为 Warp 的 Volume 对象，以便在 GPU 上进行高效重采样
    volumes = [wp.Volume.load_from_numpy(ct_images[_], bg_value=image_bgs[_]) for _ in range(2)]

    # 构建 Warp Mesh 用于距离场查询
    wp_mesh = wp.Mesh(wp.array(mesh.vertices, dtype=wp.vec3), wp.array(mesh.faces.flatten(), dtype=wp.int32))

    # 计算最大距离 (ROI 对角线长度)
    max_dist = wp.float32(np.linalg.norm(roi_size * roi_spacing))

    # 初始化输出的三通道图像张量 (通道0: 术后, 通道1: 术前, 通道2: SDF)
    image_roi = wp.full((*roi_size,), wp.vec3(image_bgs[1], image_bgs[0], wp.float32(-1.0)), wp.vec3)

    # 调用 Warp kernel 进行 GPU 加速的重采样
    wp.launch(resample_roi, image_roi.shape, [
        image_roi, origin, roi_spacing, roi_xform,
        volumes[1].id, origins[1], spacings[1], sizes[1],
        volumes[0].id, origins[0], spacings[0], sizes[0],
        post_xform, wp_mesh.id, sdf_t, max_dist,
    ])

    # 将结果转回 numpy 数组并分离术前、术后和 SDF 通道
    image_roi_np = image_roi.numpy()
    image_a, image_b, image_sdf = image_roi_np[..., 1], image_roi_np[..., 0], image_roi_np[..., 2]

    # 计算局部的简单正交物理原点
    roi_origin_itk = np.array(origin) + center
    roi_direction_itk = np.identity(3, dtype=np.float64)

    # 保存重采样后的术前、术后和 SDF 图像为 NIfTI 格式
    for op, image in (('pre', image_a), ('post', image_b), ('metal', image_sdf)):
        f = dataset_root / op / subdir / f'{prl}.nii.gz'
        f.parent.mkdir(parents=True, exist_ok=True)

        _ = itk.image_from_array(image.transpose(2, 1, 0))
        _.SetSpacing(roi_spacing)
        _.SetOrigin(roi_origin_itk)
        _.SetDirection(itk.matrix_from_array(roi_direction_itk))
        itk.imwrite(_, f.as_posix(), True)

    # 生成用于可视化的 2D 投影快照 (DRR - Digitally Reconstructed Radiograph)
    snapshot = []
    for ax in (1, 0):  # 分别在冠状面和矢状面生成投影
        stack = [
            fast_drr(image_a, ax).transpose(1, 0, 2),
            fast_drr(image_b, ax).transpose(1, 0, 2),
            fast_drr(image_sdf, ax, th=(-1.0, 1.0), mode='max').transpose(1, 0, 2)
        ]
        img = np.hstack(stack)  # 水平拼接术前、术后和SDF投影
        snapshot.append(img)

    # 垂直翻转并拼接两个视角的投影
    snapshot = np.flipud(np.hstack(snapshot))

    # 保存快照图像
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(snapshot, 'RGB').save(snapshot_file.as_posix())

    # 显式释放内存和显存资源
    del volumes
    del image_roi
    del ct_images

    import gc
    gc.collect()
    time.sleep(0.5)  # 短暂休眠以确保资源释放完毕


def launch():
    """
    主控函数：解析参数，扫描对象存储中的数据，划分数据集，并启动多进程处理。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the TOML configuration file')
    parser.add_argument('--max-workers', default=6, type=int, help='Maximum number of concurrent worker processes')
    args = parser.parse_args()

    # 加载配置并初始化 Minio 客户端
    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    # 扫描 Minio 中的 'pair' 存储桶，收集所有配对的 NIfTI 文件
    pairs = {}
    for _ in client.list_objects('pair', recursive=True):
        if not _.object_name.endswith('.nii.gz'):
            continue

        # 解析对象路径，例如: patientID/R/pre/image.nii.gz
        if len(paths := _.object_name.split('/')) != 4:
            continue

        pid, rl, op, nii = paths
        prl = f'{pid}_{rl}'

        if prl not in pairs:
            pairs[prl] = {'prl': prl}
        pairs[prl][op] = f'{pid}/{nii}'

    valid_pairs = {}
    for prl in pairs:
        try:
            # 获取上下文数据
            data = client.get_object('pair', '/'.join([prl.replace('_', '/'), 'context.toml'])).read()
            data = tomlkit.loads(data.decode('utf-8'))

            if len(data.get('excluded', [])) > 0:
                continue

            pairs[prl].update(data)

            # 获取配准元数据
            data = client.get_object('pair', '/'.join([prl.replace('_', '/'), 'align.toml'])).read()
            data = tomlkit.loads(data.decode('utf-8'))

            if len(data.get('excluded', [])) > 0:
                continue
            if 'femur_align' not in data:
                continue
            if 'hip_align' not in data:
                continue

            pairs[prl].update(data)

            # 默认标记为训练集
            pairs[prl]['is_val'] = False
            valid_pairs[prl] = pairs[prl]
        except S3Error:
            pass

    pairs = valid_pairs

    # 划分验证集 (取总数的 10%，最多 100 个)
    keys = sorted(pairs.keys())
    total = len(keys)
    n_val = min(int(total * 0.1), 100)

    if n_val > 0:
        # 均匀采样作为验证集
        for i in range(n_val):
            idx = int(i * total / n_val)
            pairs[keys[idx]]['is_val'] = True

    # 构建多进程任务列表
    tasks = [(args.config, prl, pair) for prl, pair in pairs.items()]

    # 确保单进程显存回收：使用 'spawn' 模式启动子进程，并在每个任务完成后销毁子进程 (maxtasksperchild=1)
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        # 使用 tqdm 显示进度条，无序映射任务以最大化并发效率
        for _ in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
            pass


def process(args):
    """
    多进程任务的包装函数，用于捕获并处理键盘中断异常。
    """
    try:
        main(*args)
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')


if __name__ == '__main__':
    launch()
