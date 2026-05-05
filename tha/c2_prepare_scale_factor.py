import argparse
from contextlib import nullcontext
from pathlib import Path

import define
import itk
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from torch.amp import autocast
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--subtask', required=True, type=str)
    parser.add_argument('--batch_size', default=36, type=int)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    train_root = Path(cfg['train']['root'])
    dataset_root = train_root / 'dataset'
    # cache_dir = train_root / 'cache'
    ckpt_dir = train_root / 'checkpoints'

    task = 'vae'
    (
        use_amp,
        num_workers,
        patch_size,
    ) = [
        cfg['train'][task][_]
        for _ in (
            'use_amp',
            'num_workers',
            'patch_size',
        )
    ]
    subtask = str(args.subtask)
    patch_size = list(patch_size)

    load_pt = ckpt_dir / f'{task}_{subtask}_best.pt'

    try:
        print(f'Loading checkpoint from {load_pt}...')
        checkpoint = torch.load(load_pt, map_location=device, weights_only=True)
        channels = int(checkpoint['channels'])
        print(f'Channels: {channels}')

        vae = define.vae_kl(channels).to(device)
        vae.load_state_dict(checkpoint['state_dict'])
    except Exception as e:
        raise SystemExit(f'Failed to load checkpoint: {load_pt} | {e}') from None

    val_prls, test_prls = set(cfg['val']), set(cfg['test'])
    train_files = []

    for f in (dataset_root / subtask).glob('*.nii.gz'):
        prl = '_'.join(f.name.split('_')[:2])
        if prl not in test_prls and prl not in val_prls:
            train_files.append({'image': f.as_posix()})
    print(f'Train: {len(train_files)}')

    # 使用从 checkpoint 中读取的 channels
    train_transforms = Compose(define.vae_train_transforms(patch_size, channels))

    train_ds = Dataset(train_files, train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    for _ in ('epoch', 'val_l1', 'val_psnr', 'val_ssim', 'scale_factor'):
        print(_, checkpoint.get(_))

    vae.eval()

    print('Calculating Robust Latent Scale Factor...')

    # 用于计算全局 Mean 和 Std 的统计量
    global_sum = 0.0
    global_squared_sum = 0.0
    total_elements = 0

    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        with amp_ctx:
            for batch in tqdm(train_loader):
                images = batch['image'].to(device)

                # 编码获取 Latent Mean
                z_mu, _ = vae.encode(images)
                z_flat = z_mu.detach().float()

                # 累加所有元素的和与平方和。转为 double 提取 item 彻底避免溢出
                global_sum += z_flat.sum().cpu().double().item()
                global_squared_sum += (z_flat**2).sum().cpu().double().item()
                total_elements += z_flat.numel()

    # 计算全体 Latent 元素的全局 Mean 和 Std
    global_mean = global_sum / total_elements
    global_var = (global_squared_sum / total_elements) - (global_mean**2)
    global_std = global_var**0.5

    # Stable Diffusion / LDM 标准 Scale Factor 计算：1 / 全局标准差
    scale_factor = 1.0 / global_std

    # Save scale factor
    checkpoint['scale_factor'] = scale_factor
    checkpoint['global_mean'] = global_mean
    torch.save(checkpoint, load_pt)

    print(f'Global Mean (All): {global_mean:.6f}')
    print(f'Global Std (All): {global_std:.6f}')
    print(f'Final Robust Scale Factor: {scale_factor:.6f}')

    return scale_factor


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
