import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import itk
import numpy as np
import tomlkit
import torch
from define import (
    roi_spacing,
    sdf_t,
    vae_discriminator,
    vae_kl,
    vae_perceptual_loss,
    vae_train_transforms,
    vae_val_transforms,
)
from kernel import fast_drr
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss
from monai.metrics import PSNRMetric, SSIMMetric
from monai.transforms import Compose
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--subtask', required=True)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    dataset_root = train_root / 'dataset'
    # cache_dir = train_root / 'cache'
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'vae'
    config_train = cfg['train'][task]

    subtask = args.subtask
    print('Subtask:', subtask)

    use_amp = bool(config_train['use_amp'])
    resume = bool(config_train['resume'])
    num_workers = int(config_train['num_workers'])
    num_epochs = int(config_train['num_epochs'])
    warmup_epochs = int(config_train['warmup_epochs'])
    batch_size = int(config_train['batch_size'])
    patch_size = list(config_train['patch_size'])
    sw_batch_size = int(config_train['sw_batch_size'])
    val_interval = int(config_train['val_interval'])
    adv_weight = float(config_train['adversarial_weight'])
    per_weight = float(config_train['perceptual_weight'])
    eik_weight = float(config_train['eikonal_weight'])
    kl_weight = float(config_train['kl_weight'])
    lr_g = float(config_train['lr_g'])
    lr_d = float(config_train['lr_d'])

    val_prls, test_prls = set(cfg['val']), set(cfg['test'])
    train_files, val_files, test_files = [], [], []

    for f in (dataset_root / subtask).glob('*.nii.gz'):
        prl = '_'.join(f.name.removesuffix('.nii.gz').split('_')[:2])
        if prl in test_prls:
            test_files.append({'image': f.as_posix()})
        elif prl in val_prls:
            val_files.append({'image': f.as_posix()})
        else:
            train_files.append({'image': f.as_posix()})

    print('Train:\t', len(train_files))
    print('Val:\t', len(val_files))

    # 实际通道数
    if len(train_files):
        image = itk.imread(train_files[0]['image'])
        channels = image.GetNumberOfComponentsPerPixel()
    else:
        raise RuntimeError
    print(f'Channels: {channels}')

    train_transforms = Compose(vae_train_transforms(patch_size, channels))
    val_transforms = Compose(vae_val_transforms(patch_size, channels))

    train_ds = Dataset(train_files, train_transforms)
    val_ds = Dataset(val_files, val_transforms) if len(val_files) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers) if val_ds else None

    # 生成器 (VAE)
    vae = vae_kl(channels).to(device)

    # 判别器 (PatchGAN)
    discriminator = vae_discriminator(channels).to(device)

    # L1 损失
    l1_loss_fn = torch.nn.L1Loss()

    # 对抗损失
    adv_loss_fn = PatchAdversarialLoss(criterion='least_squares')

    # 感知损失
    if subtask in ('pre',):
        per_loss_fn = vae_perceptual_loss().to(device)
    else:
        per_loss_fn = None

    # 优化器
    optimizer_g = torch.optim.Adam(vae.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

    # 混合精度 Scaler
    if use_amp:
        scaler_g = GradScaler(device=device.type)
        scaler_d = GradScaler(device=device.type)
    else:
        scaler_g = None
        scaler_d = None

    # 日志
    suffix = datetime.now().strftime(f'{task}_{subtask}_%Y%m%d_%H%M%S')
    if resume:
        suffix += '_resume'
    log_dir = log_dir / suffix
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    start_epoch = 0
    best_val_l1 = float('inf')

    # 继续训练
    if resume:
        load_pt = (ckpt_dir / f'{task}_{subtask}_last.pt').resolve()
    else:
        load_pt = None

    if load_pt and load_pt.exists():
        try:
            print('Resuming:\t', load_pt)
            checkpoint = torch.load(load_pt, map_location=device, weights_only=True)

            vae.load_state_dict(checkpoint['state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])

            if use_amp and 'scaler_g' in checkpoint and 'scaler_d' in checkpoint:
                scaler_g.load_state_dict(checkpoint['scaler_g'])
                scaler_d.load_state_dict(checkpoint['scaler_d'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_l1 = checkpoint.get('best_val_l1', float('inf'))

            print('Epoch:\t', start_epoch)
            print('L1:  \t', checkpoint['val_l1'], 'best', best_val_l1)
            print('PSNR:\t', checkpoint['val_psnr'])
            print('SSIM:\t', checkpoint['val_ssim'])
            print('Scale Factor:\t', checkpoint.get('scale_factor'))
            print('Global Mean:\t', checkpoint.get('global_mean'))
            start_epoch += 1
        except Exception as e:
            raise SystemError(f'Load failed: {e}')

    # 验证滑动窗口推理，确定性编解码
    def encode_decode_mu(inputs: torch.Tensor) -> torch.Tensor:
        return vae.decode(vae.encode(inputs)[0])

    # 训练
    for epoch in range(start_epoch, num_epochs):
        warmup = epoch < warmup_epochs

        vae.train()
        discriminator.train()

        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1
            images = batch['image'].to(device, non_blocking=True)

            # 初始化 Loss 变量，确保在后续日志记录中始终定义
            per_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
            adv_loss = torch.tensor(0.0, device=device)
            eik_loss = torch.tensor(0.0, device=device)
            loss_d = torch.tensor(0.0, device=device)

            optimizer_g.zero_grad(set_to_none=True)

            amp_ctx = autocast(device.type) if use_amp else nullcontext()
            with amp_ctx:
                # 编码获取分布参数，注意 MONAI 源码实现返回的是 sigma 不是 log var
                z_mu, z_sigma = vae.encode(images)

                # 解码
                z = vae.sampling(z_mu, z_sigma)
                reconstruction = vae.decode(z)

                # L1 重建损失
                l1_loss = l1_loss_fn(reconstruction.float(), images.float())

                if torch.isnan(l1_loss):
                    raise SystemExit('NaN in l1_loss')

            loss_g = l1_loss

            # 感知损失，退出AMP避免NaN。
            # 过滤掉几乎无变化的平坦背景区域 (std接近0)，避免 MedicalNet 内部归一化除零导致 NaN，
            # 同时也避免之前“加微量噪声”方案中，噪声被内部归一化放大成纯随机信号而严重破坏 VAE 重建质量的问题。
            img_float = images.float()

            if subtask in ('pre',) and per_loss_fn is not None:
                stds = img_float.view(img_float.shape[0], -1).std(dim=1)
                valid = stds > 1e-3

                if valid.any():
                    valid_recon = reconstruction.float()[valid]
                    valid_img = img_float[valid]
                    per_loss_val = per_loss_fn(valid_recon, valid_img)

                    if torch.isnan(per_loss_val):
                        raise SystemExit('NaN in per_loss')

                    per_loss = per_loss_val * per_weight * (valid.sum().float() / img_float.shape[0])
                    loss_g += per_loss

            with amp_ctx:
                # KL 正则化损失
                z_sigma_clamped = torch.clamp(z_sigma, min=1e-6, max=1e3)
                kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma_clamped.pow(2) - torch.log(z_sigma_clamped.pow(2)) - 1)

                if torch.isnan(kl_loss):
                    raise SystemExit('NaN in kl_loss')

                # Warmup 期间降低 KL 权重，优先保证重建精度
                current_kl_weight = kl_weight  # * 0.1 if warmup else kl_weight
                kl_loss = kl_loss * current_kl_weight

                # 生成器损失
                loss_g += kl_loss

                # 预热后补充对抗损失和程函损失
                if not warmup:
                    # 1. 对抗损失
                    out = discriminator(reconstruction.float())
                    adv_loss = adv_loss_fn(out, target_is_real=True, for_discriminator=False)

                    if torch.isnan(adv_loss):
                        raise SystemExit('NaN in adv_loss')

                    adv_loss = adv_loss * adv_weight
                    loss_g += adv_loss

                    # 2. 程函损失 (Eikonal Loss): 仅用于 Metal TSDF 任务，确保几何合理性
                    if subtask == 'metal':
                        # 计算物理坐标系下的梯度模长
                        grads = torch.gradient(reconstruction.float(), spacing=(roi_spacing,) * 3, dim=(2, 3, 4))
                        grad_norm_sq = grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2
                        grad_norm = torch.sqrt(grad_norm_sq + 1e-8)

                        target_norm = 1.0 / sdf_t
                        eik_mask = (torch.abs(images) < 0.95).float()

                        if eik_mask.sum() > 0:
                            eik_loss = torch.sum(eik_mask * (grad_norm - target_norm) ** 2) / (eik_mask.sum() + 1e-8)
                            loss_g += eik_loss * eik_weight

            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer_g.step()

            # 预热后训练判别器
            if not warmup:
                optimizer_d.zero_grad(set_to_none=True)

                amp_ctx = autocast(device.type) if use_amp else nullcontext()
                with amp_ctx:
                    # Real
                    logits_real = discriminator(images.detach())
                    loss_d_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                    # Fake
                    logits_fake = discriminator(reconstruction.detach())
                    loss_d_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                    # Total Discriminator Loss
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                if use_amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()
                epoch_loss_d += loss_d.item()

            # 轮次累计
            epoch_loss_g += loss_g.item()

            # 日志
            global_step = epoch * len(train_loader) + step

            postfix = {
                'L1': f'{l1_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Per.': f'{per_loss.item():.4f}',
                'Eik.': f'{eik_loss.item():.4f}',
                'Adv.': f'{adv_loss.item():.4f}',
            }

            if step % 10 == 0:
                writer.add_scalar('train/loss_g', loss_g.item(), global_step)
                writer.add_scalar('train/loss_d', loss_d.item(), global_step)
                writer.add_scalar('train/l1_loss', l1_loss.item(), global_step)
                writer.add_scalar('train/kl_loss', kl_loss.item(), global_step)
                writer.add_scalar('train/per_loss', per_loss.item(), global_step)
                writer.add_scalar('train/eik_loss', eik_loss.item(), global_step)
                writer.add_scalar('train/adv_loss', adv_loss.item(), global_step)
                writer.add_scalar('latent/z_mu_mean', z_mu.mean().item(), global_step)
                writer.add_scalar('latent/z_sigma_mean', z_sigma.mean().item(), global_step)

            pbar.set_postfix(postfix)

        writer.add_scalar('train/epoch_loss_g', epoch_loss_g / step, epoch)

        # 验证
        if val_loader and epoch % val_interval == 0:
            # Clear cache before validation to provide maximum headroom
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            vae.eval()
            val_l1_loss = 0.0
            psnr_metric = PSNRMetric(max_val=2.0)
            ssim_metric = SSIMMetric(data_range=2.0, spatial_dims=3)

            val_pbar = tqdm(val_loader, desc='Val')

            val_step = 0
            with torch.no_grad():
                amp_ctx = autocast(device.type) if use_amp else nullcontext()
                for i, batch in enumerate(val_pbar):
                    val_step += 1
                    val_images = batch['image'].to(device, non_blocking=True)

                    with amp_ctx:
                        val_recon: torch.Tensor = sliding_window_inference(
                            inputs=val_images,
                            roi_size=patch_size,
                            sw_batch_size=sw_batch_size,
                            predictor=encode_decode_mu,
                            overlap=0.25,
                            mode='gaussian',
                            device=device,
                            sw_device=device,
                            progress=False,
                        )  # type:ignore

                    v_l1 = l1_loss_fn(val_recon.float(), val_images.float())
                    val_l1_loss += v_l1.item()

                    psnr_metric(y_pred=val_recon, y=val_images)
                    ssim_metric(y_pred=val_recon, y=val_images)

                    val_pbar.set_postfix({'L1': f'{v_l1.item():.4f}'})

                    # 可视化
                    if i == 0:
                        vis_inputs, vis_recons, vis_diffs = [], [], []
                        axis = 1  # 冠状面投影

                        for c in range(channels):
                            img_in = val_images[0, c].cpu().numpy()
                            img_rec = val_recon[0, c].cpu().numpy()

                            # 输入 DRR
                            drr_in = fast_drr(img_in + 1.0, axis, th=(0.0, 2.0), mode='mean')
                            drr_in = np.flipud(drr_in.transpose(1, 0, 2))
                            vis_inputs.append(drr_in)

                            # 重建 DRR
                            drr_rec = fast_drr(img_rec + 1.0, axis, th=(0.0, 2.0), mode='mean')
                            drr_rec = np.flipud(drr_rec.transpose(1, 0, 2))
                            vis_recons.append(drr_rec)

                            # 差异 DRR
                            diff = np.abs(img_in - img_rec)
                            drr_diff = fast_drr(diff, axis, th=(0.0, 1.0), mode='max')
                            drr_diff = np.flipud(drr_diff.transpose(1, 0, 2))
                            vis_diffs.append(drr_diff)

                        # 拼接通道
                        vis_input = np.hstack(vis_inputs)
                        vis_recon = np.hstack(vis_recons)
                        vis_diff = np.hstack(vis_diffs)

                        writer.add_image('val/Input', vis_input, epoch, dataformats='HWC')
                        writer.add_image('val/Recon', vis_recon, epoch, dataformats='HWC')
                        writer.add_image('val/Diff', vis_diff, epoch, dataformats='HWC')

                        val_vis_dir = log_dir / 'val'
                        val_vis_dir.mkdir(parents=True, exist_ok=True)

                        Image.fromarray(vis_input).save(val_vis_dir / f'{epoch:04d}_input.png')
                        Image.fromarray(vis_recon).save(val_vis_dir / f'{epoch:04d}_recon.png')
                        Image.fromarray(vis_diff).save(val_vis_dir / f'{epoch:04d}_diff.png')

            val_l1_loss /= val_step
            psnr = psnr_metric.aggregate().item()
            ssim = ssim_metric.aggregate().item()
            psnr_metric.reset()
            ssim_metric.reset()

            writer.add_scalar('val/l1', val_l1_loss, epoch)
            writer.add_scalar('val/psnr', psnr, epoch)
            writer.add_scalar('val/ssim', ssim, epoch)

            print(f'Val Epoch {epoch}: L1={val_l1_loss:.5f} | PSNR={psnr:.4f} | SSIM={ssim:.4f}')

            # 保存
            checkpoint = {
                'epoch': epoch,
                'channels': channels,
                'state_dict': vae.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'val_l1': val_l1_loss,
                'val_psnr': psnr,
                'val_ssim': ssim,
                'best_val_l1': best_val_l1,
            }

            if use_amp:
                checkpoint['scaler_g'] = scaler_g.state_dict()
                checkpoint['scaler_d'] = scaler_d.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if val_l1_loss < best_val_l1:  # 不用 PSNR 或 SSIM，L1 最能反映骨质宏观占位和均值偏移，同时也最能确保 TSDF 准确
                best_val_l1 = val_l1_loss
                checkpoint['best_val_l1'] = best_val_l1

                torch.save(checkpoint, ckpt_dir / f'{task}_{subtask}_best.pt')
                print('New best model saved!')

            torch.save(checkpoint, ckpt_dir / f'{task}_{subtask}_last.pt')

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
