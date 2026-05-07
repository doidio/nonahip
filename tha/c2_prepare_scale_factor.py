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
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


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
        prl = '_'.join(f.name.split('_')[:2])
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
        m_name = 'FID'
    else:
        r_val = sum(all_recon_metrics) / N
        i_val = sum(all_interp_metrics) / N
        m_name = 'Eikonal'

    ratio = i_val / (r_val + 1e-12)

    # Save
    checkpoint['scale_factor'] = float(scale_factor)
    checkpoint['global_mean'] = float(global_mean)
    checkpoint[f'r{m_name.lower()}'] = float(r_val)
    checkpoint[f'i{m_name.lower()}'] = float(i_val)
    torch.save(checkpoint, load_pt)

    print(f'\n--- Summary for {subtask} ---')
    print(f'Scale Factor: {scale_factor:.6f}')
    print(f'r{m_name} (Reconstruction): {r_val:.6f}')
    print(f'i{m_name} (Interpolation):  {i_val:.6f}')
    print(f'Ratio (i/r): {ratio:.2f}x')

    if ratio > 10.0:
        print('>> WARNING: High Interpolation Gap! Latent space might be fragmented.')
    else:
        print('>> SUCCESS: Low Interpolation Gap. Latent space is continuous.')

    return scale_factor, i_val


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
