import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, SaveImage
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import define

try:
    import torch_musa

    device = torch.device('musa' if torch.musa.is_available() else 'cpu')  # noqa
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = tomlkit.loads(config_path.read_text('utf-8')).unwrap()

    train_root = Path(str(config['train']['root']))
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'ldm'
    (
        use_amp, resume, num_workers, num_epochs, val_interval, val_limit,
        batch_size, sw_batch_size, lr, gradient_accumulation_steps, ema_decay, cfg,
    ) = [config['train'][task][_] for _ in (
        'use_amp', 'resume', 'num_workers', 'num_epochs', 'val_interval', 'val_limit',
        'batch_size', 'sw_batch_size', 'lr', 'gradient_accumulation_steps', 'ema_decay', 'cfg',
    )]

    # 既然每个 batch 处理多张图，梯度累积相应减少
    gradient_accumulation_steps = max(1, gradient_accumulation_steps // batch_size)
    print('List Batch Size:\t', batch_size)
    print('Grad Accu Steps:\t', gradient_accumulation_steps)

    patch_size = list(config['train']['vae']['patch_size'])

    train_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'train').glob('*.npy'))]
    val_files = [{'image': p.as_posix()} for p in sorted((train_root / 'latents' / 'val').glob('*.npy'))]
    print('Train:\t', len(train_files))
    print('Val:\t', len(val_files))

    val_total = min(val_limit, len(val_files))
    if val_total > 1:
        val_files = val_files[::max(1, len(val_files) // (val_total - 1))]
    print('Val limited:\t', len(val_files))

    def load_vae(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()

        print(f'[{subtask}]\t', f'Loading {ckpt_path}')

        loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
        vae_model = define.vae_kl().to(device)
        vae_model.load_state_dict(loaded['state_dict'])
        vae_model.eval().float()

        print('Epoch:\t', loaded['epoch'])
        print('L1:   \t', loaded['val_l1'], 'best', loaded['best_val_l1'])
        print('PSNR:\t', loaded['val_psnr'])
        print('SSIM:\t', loaded['val_ssim'])
        print('Scale Factor:\t', sf := loaded['scale_factor'])
        print('Global Mean:\t', mean := loaded['global_mean'])

        return vae_model, sf, mean

    vae_image, image_sf, image_mean = load_vae('metal')
    vae_cond, cond_sf, cond_mean = load_vae('pre')

    transforms = Compose(define.ldm_transforms(
        image_mean=image_mean, image_sf=image_sf,
        cond_mean=cond_mean, cond_sf=cond_sf,
    ))

    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    # 训练 Loader 使用 custom_collate
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
        collate_fn=define.ldm_collate_fn,
    )
    # 验证 Loader 保持 BS=1 即可
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    ldm = define.ldm_unet().to(device)
    ema = define.EMA(ldm, decay=ema_decay)

    scheduler = define.scheduler_ddpm()
    num_train_timesteps = scheduler.num_train_timesteps

    optimizer = torch.optim.AdamW(ldm.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    ldm_ckpt_path = (ckpt_dir / f'{task}_last.pt').resolve()

    if resume and ldm_ckpt_path.exists():
        try:
            print('Resuming:\t', ldm_ckpt_path)
            ckpt = torch.load(ldm_ckpt_path, map_location=device)
            ldm.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if 'ema_state' in ckpt:
                ema.load_state_dict(ckpt['ema_state'])

            if use_amp and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            start_epoch = ckpt['epoch'] + 1
            val_loss = ckpt.get('val_loss', float('inf'))

            print('Epoch:\t', start_epoch)
            print('MSE:\t', val_loss)
        except Exception as e:
            print(f'Load failed: {e}')

    suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
    if resume:
        suffix += '_resume'
    log_dir = log_dir / suffix
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    saver = SaveImage(
        output_dir=log_dir,
        output_postfix='',
        output_ext='.nii.gz',
        separate_folder=False,
        print_log=False,
        resample=False,
    )

    def decode_gpu(z, name, vae_model, sf, mean, ep):
        z = (z / sf + mean).detach().to(device).float()
        with torch.no_grad():
            recon = define.vae_decode_tiled(
                z=z, vae=vae_model, patch_size=patch_size, sw_batch_size=sw_batch_size,
            )
        name = f'val_epoch_{ep:03d}_{name}.nii.gz'
        saver(recon[0].cpu(), meta_data={'filename_or_obj': name})
        return recon.cpu()

    amp_ctx = autocast(device.type) if use_amp else nullcontext()

    for epoch in range(start_epoch, num_epochs):
        ldm.train()
        epoch_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1

            # batch 现在是一个 dict，里面的 'image' 和 'condition' 是 List[Tensor]
            image_list = batch['image']
            cond_list = batch['condition']

            current_bs = len(image_list)

            # 累加这个 list 里所有的 loss
            total_loss_for_this_batch = torch.tensor(0.0, device=device)

            with amp_ctx:
                for b_idx in range(current_bs):
                    # 取出当前单张样本，增加 Batch 维度使其变成 [1, C, D, H, W]
                    image = image_list[b_idx].unsqueeze(0).to(device, non_blocking=True)
                    cond = cond_list[b_idx].unsqueeze(0).to(device, non_blocking=True)

                    # CFG Condition Dropout
                    drop_mask = (torch.rand(1, 1, 1, 1, 1, device=device) < 0.15).float()
                    cond = cond * (1.0 - drop_mask)

                    # 采样时间步 t
                    timesteps = torch.randint(0, num_train_timesteps, (1,), device=device).long()

                    # 生成噪声
                    noise = torch.randn_like(image)

                    # 加噪过程
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                    # 拼接输入
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    # 预测噪声
                    noise_pred = ldm(x=input_tensor, timesteps=timesteps)

                    # 计算这一个样本的损失
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                    # 累加，除以 current_bs 相当于在 List 内部求了平均
                    # 除以 gradient_accumulation_steps 是为了跨 step 累积
                    total_loss_for_this_batch += (loss / current_bs / gradient_accumulation_steps)

            # 反向传播 (把这个 List 里所有图的计算图一起 backward)
            if use_amp:
                scaler.scale(total_loss_for_this_batch).backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)
                    ema.update(ldm)
            else:
                total_loss_for_this_batch.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(ldm.parameters(), 1.0)
                    optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    ema.update(ldm)

            # 这里记录的是当前这一步的平均 MSE 损失，用于显示
            display_loss = (total_loss_for_this_batch.item() * gradient_accumulation_steps)
            epoch_loss += display_loss

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss', display_loss, global_step)

            pbar.set_postfix({'MSE': f'{display_loss:.4f}'})

        writer.add_scalar('train/epoch_loss', epoch_loss / step, epoch)

        # 验证与采样 (保持 BS=1，不需要改 collate_fn)
        if epoch % val_interval == 0:
            ldm.eval()
            ema.store(ldm)
            ema.copy_to(ldm)

            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for i, batch in enumerate(val_bar := tqdm(val_loader, desc='Val')):
                    image = batch['image'].to(device)
                    cond = batch['condition'].to(device)

                    timesteps = torch.randint(0, num_train_timesteps, (image.shape[0],), device=device).long()
                    noise = torch.randn_like(image)
                    noisy_image = scheduler.add_noise(image, noise, timesteps)
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    with amp_ctx:
                        noise_pred = ldm(input_tensor, timesteps)
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                    val_loss += loss.item()
                    val_steps += 1

                    if i == 0:
                        val_scheduler = define.scheduler_ddim()
                        val_scheduler.set_timesteps(num_inference_steps=50, device=device)

                        generator = torch.Generator(device=device).manual_seed(42)
                        generated = torch.randn(image.shape, device=device, generator=generator)

                        for t in val_scheduler.timesteps:
                            val_bar.set_postfix({'DDIM': t.item()})
                            latent_input = torch.cat([generated] * 2)
                            uncond = torch.zeros_like(cond)
                            cond_input = torch.cat([cond, uncond])
                            model_input = torch.cat([latent_input, cond_input], dim=1)

                            with torch.no_grad():
                                t_input = t[None].to(device).repeat(2)
                                noise_pred_batch = ldm(model_input, t_input)

                            noise_cond, noise_uncond = noise_pred_batch.chunk(2)
                            noise_pred = noise_uncond + cfg * (noise_cond - noise_uncond)

                            with torch.no_grad():
                                generated, _ = val_scheduler.step(noise_pred, t, generated)

                        with amp_ctx:
                            vis_generated = decode_gpu(generated, 'Generated', vae_image, image_sf, image_mean, epoch)
                            if epoch == 0:
                                vis_gt = decode_gpu(image, 'GroundTruth', vae_image, image_sf, image_mean, epoch)
                                vis_cond = decode_gpu(cond, 'Condition', vae_cond, cond_sf, cond_mean, epoch)

                        def norm_vis(x):
                            x_min, x_max = x.min(), x.max()
                            return (x - x_min) / (x_max - x_min + 1e-5)

                        y_idx = vis_generated.shape[3] // 2  # [B, C, Z, Y, X]

                        vis_generated_slice = vis_generated[0, 0, :, y_idx, :] # [Z, X]
                        writer.add_image('val/Generated', norm_vis(vis_generated_slice), epoch, dataformats='HW')

                        if epoch == 0:
                            vis_cond_slice = vis_cond[0, 0, :, y_idx, :]
                            vis_gt_slice = vis_gt[0, 0, :, y_idx, :]
                            writer.add_image('val/Condition', norm_vis(vis_cond_slice), epoch, dataformats='HW')
                            writer.add_image('val/GroundTruth', norm_vis(vis_gt_slice), epoch, dataformats='HW')

            ema.restore(ldm)
            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print('Val Loss:\t', avg_val_loss)

            ckpt = {
                'epoch': epoch,
                'state_dict': ldm.state_dict(),
                'ema_state': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if epoch % 50 == 0:
                torch.save(ckpt, ckpt_dir / f'{task}_{epoch:03d}.pt')
                print(f'Model saved at epoch {epoch}!')

            torch.save(ckpt, ckpt_dir / f'{task}_last.pt')

        torch.cuda.empty_cache()

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
