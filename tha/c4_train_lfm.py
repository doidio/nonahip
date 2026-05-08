import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import define
import numpy as np
import tomlkit
import torch
from kernel import fast_drr
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SaveImage
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = tomlkit.loads(config_path.read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'lfm'
    (
        use_amp,
        resume,
        num_workers,
        num_epochs,
        val_interval,
        batch_size,
        sw_batch_size,
        lr,
        gradient_accumulation_steps,
        ema_decay,
        classifier_free_guidence,
    ) = [
        cfg['train'][task][_]
        for _ in (
            'use_amp',
            'resume',
            'num_workers',
            'num_epochs',
            'val_interval',
            'batch_size',
            'sw_batch_size',
            'lr',
            'gradient_accumulation_steps',
            'ema_decay',
            'classifier_free_guidence',
        )
    ]

    # 既然每个 batch 处理多张图，梯度累积相应减少
    gradient_accumulation_steps = max(1, gradient_accumulation_steps // batch_size)
    print('List Batch Size:\t', batch_size)
    print('Grad Accu Steps:\t', gradient_accumulation_steps)

    patch_size = list(cfg['train']['vae']['patch_size'])

    val_prls, test_prls = set(cfg['val'].keys()), set(cfg['test'].keys())
    train_files, val_files, test_files = [], [], []

    for f in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(f.name.removesuffix('.npy').split('_')[:2])
        if prl in cfg['pairs']['excluded']:
            continue
        if prl in test_prls:
            test_files.append({'image': f.as_posix(), 'prl': prl})
        elif prl in val_prls:
            val_files.append({'image': f.as_posix(), 'prl': prl})
        else:
            train_files.append({'image': f.as_posix(), 'prl': prl})

    val_prl = val_files[0]['prl'] if len(val_files) else None

    print('Train:\t', len(train_files))
    print('Val:\t', len(val_files))

    def load_vae(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()

        print(f'[{subtask}]\t', f'Loading {ckpt_path}')

        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        channels = loaded['channels']
        vae_model = define.vae_kl(channels).to(device)
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

    transforms = Compose(
        define.lfm_transforms(
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        )
    )

    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    # 训练 Loader 使用 custom_collate
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=define.lfm_collate_fn,
    )
    # 验证 Loader 保持 BS=1 即可
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    lfm = define.lfm_unet().to(device)
    ema = define.EMA(lfm, decay=ema_decay)

    scheduler = define.scheduler_rflow()

    optimizer = torch.optim.AdamW(lfm.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    lfm_ckpt_path = (ckpt_dir / f'{task}_last.pt').resolve()

    if resume and lfm_ckpt_path.exists():
        try:
            print('Resuming:\t', lfm_ckpt_path)
            ckpt = torch.load(lfm_ckpt_path, map_location=device)
            lfm.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if 'ema_state' in ckpt:
                ema.load_state_dict(ckpt['ema_state'])

            if use_amp and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            start_epoch = ckpt['epoch']
            val_loss = ckpt.get('val_loss', float('inf'))

            print('Epoch:\t', start_epoch)
            print('MSE:\t', val_loss)
            start_epoch += 1
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

    def decode(z, name, vae_model, sf, mean, ep):
        z = (z / sf + mean).detach().to(device).float()

        def decode_predictor(inputs: torch.Tensor) -> torch.Tensor:
            vae_latent_ch = vae_model.latent_channels
            if inputs.shape[1] > vae_latent_ch:
                recons = []
                for i in range(0, inputs.shape[1], vae_latent_ch):
                    recons.append(vae_model.decode(inputs[:, i : i + vae_latent_ch]))
                return torch.cat(recons, dim=1)
            return vae_model.decode(inputs)

        with torch.no_grad():
            recon = sliding_window_inference(
                inputs=z,
                roi_size=[p // define.vae_downsample for p in patch_size],
                sw_batch_size=sw_batch_size,
                predictor=decode_predictor,
                overlap=0.25,
                mode='gaussian',
                device=device,
                sw_device=device,
                progress=False,
            )

        name = f'val_epoch_{ep:03d}_{name}.nii.gz'
        saver(recon[0].cpu(), meta_data={'filename_or_obj': name})
        return recon.cpu()

    amp_ctx = autocast(device.type) if use_amp else nullcontext()

    for epoch in range(start_epoch, num_epochs):
        lfm.train()
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

                    # 采样时间步
                    timesteps = scheduler.sample_timesteps(image)

                    # 生成噪声
                    noise = torch.randn_like(image)

                    # RFM 加噪过程
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                    # 拼接输入
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    # 预测速度 (Velocity)
                    velocity_pred = lfm(x=input_tensor, timesteps=timesteps)

                    # RFM 的目标是真实数据与噪声的差: target_velocity = image - noise
                    target_velocity = image - noise

                    # 计算这一个样本的损失
                    loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float())

                    # 累加，除以 current_bs 相当于在 List 内部求了平均
                    # 除以 gradient_accumulation_steps 是为了跨 step 累积
                    total_loss_for_this_batch += loss / current_bs / gradient_accumulation_steps

            # 反向传播 (把这个 List 里所有图的计算图一起 backward)
            if use_amp:
                scaler.scale(total_loss_for_this_batch).backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lfm.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)
                    ema.update(lfm)
            else:
                total_loss_for_this_batch.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(lfm.parameters(), 1.0)
                    optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    ema.update(lfm)

            # 这里记录的是当前这一步的平均 MSE 损失，用于显示
            display_loss = total_loss_for_this_batch.item() * gradient_accumulation_steps
            epoch_loss += display_loss

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss', display_loss, global_step)

            pbar.set_postfix({'MSE': f'{display_loss:.4f}'})

        writer.add_scalar('train/epoch_loss', epoch_loss / step, epoch)

        # 验证与采样 (保持 BS=1，不需要改 collate_fn)
        if epoch % val_interval == 0:
            lfm.eval()
            ema.store(lfm)
            ema.copy_to(lfm)

            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for i, batch in enumerate(val_bar := tqdm(val_loader, desc='Val')):
                    image = batch['image'].to(device)
                    cond = batch['condition'].to(device)

                    timesteps = scheduler.sample_timesteps(image)
                    noise = torch.randn_like(image)
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    with amp_ctx:
                        velocity_pred = lfm(input_tensor, timesteps)
                        target_velocity = image - noise
                        loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float())

                    val_loss += loss.item()
                    val_steps += 1

                    prl = batch['prl'][0]
                    if prl == val_prl:
                        name = f'{prl}_{i}'

                        num_inference_steps = 50
                        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
                        all_timesteps = scheduler.timesteps
                        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

                        generator = torch.Generator(device=device).manual_seed(42)
                        generated = torch.randn(image.shape, device=device, generator=generator)

                        for t, next_t in zip(all_timesteps, all_next_timesteps):
                            val_bar.set_postfix({'RFlow': t.item()})

                            latent_input = torch.cat([generated] * 2)
                            uncond = torch.zeros_like(cond)
                            cond_input = torch.cat([cond, uncond])
                            model_input = torch.cat([latent_input, cond_input], dim=1)

                            with torch.no_grad():
                                t_input = t[None].to(device).repeat(2)
                                velocity_pred_batch = lfm(model_input, t_input)

                            velocity_cond, velocity_uncond = velocity_pred_batch.chunk(2)
                            velocity_pred = velocity_uncond + classifier_free_guidence * (velocity_cond - velocity_uncond)

                            with torch.no_grad():
                                generated, _ = scheduler.step(velocity_pred, t, generated, next_t)

                        with amp_ctx:
                            vis_generated = decode(generated, f'{name}_val_epoch_{epoch:03d}_Gen', vae_image, image_sf, image_mean, epoch)
                            vis_gt = decode(image, f'{name}_val_epoch_{epoch:03d}_GT', vae_image, image_sf, image_mean, epoch)
                            vis_cond = decode(cond, f'{name}_val_epoch_{epoch:03d}_Cond', vae_cond, cond_sf, cond_mean, epoch)

                        # DRR Visualization (Refer to VAE style)
                        axis = 1
                        val_vis_dir = log_dir / 'val'
                        val_vis_dir.mkdir(parents=True, exist_ok=True)

                        def get_drr_hstack(vis_tensor):
                            drrs = []
                            for c in range(vis_tensor.shape[1]):
                                img = vis_tensor[0, c].numpy()
                                drr = fast_drr(img + 1.0, axis, th=(0.0, 2.0), mode='mean')
                                drrs.append(np.flipud(drr.transpose(1, 0, 2)))
                            return np.hstack(drrs)

                        drr_gen = get_drr_hstack(vis_generated)
                        drr_gt = get_drr_hstack(vis_gt)
                        drr_cond = get_drr_hstack(vis_cond)

                        writer.add_image(f'val/{name}_Gen', drr_gen, epoch, dataformats='HWC')
                        writer.add_image(f'val/{name}_GT', drr_gt, epoch, dataformats='HWC')
                        writer.add_image(f'val/{name}_Cond', drr_cond, epoch, dataformats='HWC')

                        Image.fromarray(drr_gen).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_Gen.png')
                        Image.fromarray(drr_gt).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_GT.png')
                        Image.fromarray(drr_cond).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_Cond.png')

                        # Diff DRR (hstack)
                        diff_drrs = []
                        for c in range(vis_generated.shape[1]):
                            diff = np.abs(vis_generated[0, c].numpy() - vis_gt[0, c].numpy())
                            drr_diff = fast_drr(diff, axis, th=(0.0, 1.0), mode='mean')
                            diff_drrs.append(np.flipud(drr_diff.transpose(1, 0, 2)))

                        drr_diff_hstack = np.hstack(diff_drrs)
                        writer.add_image(f'val/Diff_{i}', drr_diff_hstack, epoch, dataformats='HWC')
                        Image.fromarray(drr_diff_hstack).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_Diff.png')

            ema.restore(lfm)
            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print('Val Loss:\t', avg_val_loss)

            ckpt = {
                'epoch': epoch,
                'state_dict': lfm.state_dict(),
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
