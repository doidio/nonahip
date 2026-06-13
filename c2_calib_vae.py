import argparse
from contextlib import nullcontext
from pathlib import Path

import define
import numpy as np
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from scipy import linalg
from torch.amp import autocast
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance with stability improvements."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # 1. 稳定性优化：主动添加 epsilon 确保矩阵正定，消除奇异性警告
    offset = np.eye(sigma1.shape[0]) * eps
    s1, s2 = sigma1 + offset, sigma2 + offset

    # 2. 算法优化：利用对称性计算 Tr(sqrt(S1 @ S2))
    # 这种方法比通用的 linalg.sqrtm 更稳定，且避免了已弃用的 'disp' 参数
    try:
        # 计算 S1 的平方根
        w1, v1 = linalg.eigh(s1)
        s1_sqrt = v1 @ np.diag(np.sqrt(np.maximum(w1, 0))) @ v1.T

        # 计算对称乘积 sqrt(S1) @ S2 @ sqrt(S1) 的平方根特征值
        m = s1_sqrt @ s2 @ s1_sqrt
        wm, _ = linalg.eigh(m)
        tr_covmean = np.sum(np.sqrt(np.maximum(wm, 0)))
    except (ValueError, linalg.LinAlgError):
        # 极端情况下的兜底方案，移除 disp 参数以保持兼容
        covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--subtask', required=True, type=str, choices=['pre', 'metal'])
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()

    train_root = Path(cfg['train']['root'])
    dataset_root = train_root / 'dataset'
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
        checkpoint = torch.load(load_pt, map_location=device, weights_only=False)
        channels = int(checkpoint['channels'])
        print(f'Channels: {channels}')

        vae = define.vae_kl(channels).to(device)
        vae.load_state_dict(checkpoint['state_dict'])
        vae.eval()
    except Exception as e:
        raise SystemExit(f'Failed to load checkpoint: {load_pt} | {e}') from None

    # 初始化特征提取器 (用于 pre 任务)
    feature_extractor = None
    print('Initializing Feature Extractor (MedicalNet)...')
    perceptual_loss = define.vae_perceptual_loss().to(device)
    perceptual_loss.eval()
    feature_extractor = perceptual_loss.perceptual_function.model

    val_prls, test_prls = set(cfg['val']), set(cfg['test'])
    train_files = []

    for f in (dataset_root / subtask).glob('*.nii.gz'):
        prl = '_'.join(f.name.removesuffix('.nii.gz').split('_')[:2])
        if prl in cfg['pairs']['excluded']:
            continue
        if prl not in test_prls and prl not in val_prls:
            train_files.append({'image': f.as_posix()})
    print(f'Train: {len(train_files)}')

    train_transforms = Compose(define.vae_train_transforms(patch_size, channels))
    train_ds = Dataset(train_files, train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print('Phase 1: Collecting Latents and Real Features...')
    all_z = []
    all_real_features = []

    global_sum = 0.0
    global_squared_sum = 0.0
    total_elements = 0

    with torch.no_grad():
        amp_ctx = autocast(device.type) if use_amp else nullcontext()
        for batch in tqdm(train_loader, desc='Encoding'):
            images = batch['image'].to(device)

            with amp_ctx:
                # 1. Scale Factor
                z_mu, _ = vae.encode(images)
                z_flat = z_mu.detach().float()
                global_sum += z_flat.sum().cpu().double().item()
                global_squared_sum += (z_flat**2).sum().cpu().double().item()
                total_elements += z_flat.numel()

                all_z.append(z_mu.detach().cpu().as_tensor() if hasattr(z_mu, 'as_tensor') else z_mu.detach().cpu())

                # 2. Real Features
                real_feat = feature_extractor(images.float())
                if hasattr(real_feat, 'as_tensor'):
                    real_feat = real_feat.as_tensor()
                if real_feat.ndim > 2:
                    real_feat = torch.nn.functional.adaptive_avg_pool3d(real_feat, 1).flatten(1)
                all_real_features.append(real_feat.cpu())

    global_mean = global_sum / total_elements
    global_var = (global_squared_sum / total_elements) - (global_mean**2)
    global_std = global_var**0.5
    scale_factor = 1.0 / global_std

    z = torch.cat(all_z, dim=0)
    N = z.shape[0]
    z_flatten = z.view(N, -1).to(device)

    print('Phase 2: Finding Nearest Neighbors in Latent Space...')
    nn_indices = []
    chunk_size = 512
    for i in range(0, N, chunk_size):
        z_chunk = z_flatten[i : i + chunk_size]
        dists = torch.cdist(z_chunk, z_flatten, p=2)
        for j in range(len(z_chunk)):
            dists[j, i + j] = float('inf')
        nn_indices.append(dists.argmin(dim=1).cpu())

    nn_indices = torch.cat(nn_indices, dim=0)

    print('Phase 3: Decoding Reconstruction & Interpolation...')
    all_recon_metrics = []
    all_interp_metrics = []

    for i in tqdm(range(0, N, args.batch_size), desc='Evaluating'):
        idx = list(range(i, min(i + args.batch_size, N)))
        z_batch = z[idx].to(device)
        z_nn = z[nn_indices[idx]].to(device)
        z_interp = 0.5 * z_batch + 0.5 * z_nn

        with torch.no_grad():
            with amp_ctx:
                # 重建与插值解码
                recon_img = vae.decode(z_batch)
                interp_img = vae.decode(z_interp)

                if subtask == 'pre':
                    # iFID / rFID 特征提取
                    def get_feat(img):
                        feat = feature_extractor(img.float())
                        if hasattr(feat, 'as_tensor'):
                            feat = feat.as_tensor()
                        if feat.ndim > 2:
                            feat = torch.nn.functional.adaptive_avg_pool3d(feat, 1).flatten(1)
                        return feat.cpu()

                    all_recon_metrics.append(get_feat(recon_img))
                    all_interp_metrics.append(get_feat(interp_img))

                elif subtask == 'metal':
                    # iEikonal / rEikonal 计算
                    def get_eik(img):
                        grads = torch.gradient(img.float(), spacing=(define.roi_spacing,) * 3, dim=(2, 3, 4))
                        grad_norm = torch.sqrt(grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2 + 1e-8)
                        target_norm = 1.0 / define.sdf_t
                        mask = (torch.abs(img.float()) < 0.95).float()
                        if mask.sum() > 0:
                            return (torch.sum(mask * (grad_norm - target_norm) ** 2) / (mask.sum() + 1e-8)).cpu().item()
                        return 0.0

                    all_recon_metrics.append(get_eik(recon_img) * len(idx))
                    all_interp_metrics.append(get_eik(interp_img) * len(idx))

    # Phase 4: Summary Calculation
    real_feats = torch.cat(all_real_features, dim=0).numpy()
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)

    print('Phase 4: Summary...')
    if subtask == 'pre':
        recon_feats = torch.cat(all_recon_metrics, dim=0).numpy()
        interp_feats = torch.cat(all_interp_metrics, dim=0).numpy()

        r_val = calculate_frechet_distance(mu_real, sigma_real, np.mean(recon_feats, 0), np.cov(recon_feats, rowvar=False))
        i_val = calculate_frechet_distance(mu_real, sigma_real, np.mean(interp_feats, 0), np.cov(interp_feats, rowvar=False))
        m_name = 'FID (特征分布距离)'
        # MedicalNet FID 阈值设定
        threshold_excellent = 0.05
        threshold_warn = 0.5

    else:  # subtask == 'metal'
        r_val = sum(all_recon_metrics) / N
        i_val = sum(all_interp_metrics) / N
        m_name = 'Eikonal (距离场梯度MSE)'
        # TSDF Eikonal 阈值设定
        threshold_excellent = 0.01
        threshold_warn = 0.05

    # 计算绝对差值 (Gap)
    delta = i_val - r_val

    # Save
    checkpoint['scale_factor'] = float(scale_factor)
    checkpoint['global_mean'] = float(global_mean)
    checkpoint[f'r{m_name.split()[0].lower()}'] = float(r_val)
    checkpoint[f'i{m_name.split()[0].lower()}'] = float(i_val)
    torch.save(checkpoint, load_pt)

    print(f'\n--- Summary for {subtask} ---')
    print(f'Epoch: \t\t {checkpoint.get("epoch", "N/A")}')
    print(f'L1:    \t\t {checkpoint.get("val_l1", 0):.6f} (Best: {checkpoint.get("best_val_l1", 0):.6f})')
    print(f'MSE:  \t\t {checkpoint.get("val_mse", 0):.6f}')
    print(f'PSNR:  \t\t {checkpoint.get("val_psnr", 0):.2f}')
    print(f'SSIM:  \t\t {checkpoint.get("val_ssim", 0):.4f}')
    print('-' * 30)
    print(f'Scale Factor:\t {scale_factor:.6f}')
    print(f'Global Mean:\t {global_mean:.6f}')
    print(f'r{m_name.split()[0]} (Recon):\t {r_val:.6f}  <- 越接近 0 模型越好')
    print(f'i{m_name.split()[0]} (Interp):\t {i_val:.6f}  <- 绝对插值表现')
    print(f'Gap (i - r): \t {delta:.6f}  <- 插值造成的性能退化量')
    print('-' * 30)

    if delta < threshold_excellent:
        print(f'>> Δ{m_name.split()[0]} < {threshold_excellent} 隐空间高度平滑连续')
    elif delta < threshold_warn:
        print(f'>> Δ{threshold_excellent} <= {m_name.split()[0]} < {threshold_warn} 隐空间插值性能下降')
    else:
        print(f'>> Δ{m_name.split()[0]} >= {threshold_warn} 隐空间严重碎片化')

    return scale_factor, i_val


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
