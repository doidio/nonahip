import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import define
import numpy as np
import tomlkit
import torch
from kernel import fast_drr
from monai.data import DataLoader, Dataset, pad_list_data_collate
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
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = tomlkit.loads(config_path.read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    dataset_root = Path(cfg['dataset']['root'])
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'rflow'
    (
        use_amp,
        num_workers,
        num_epochs,
        val_interval,
        batch_size,
        sw_batch_size,
        lr,
        effective_batch_size,
        ema_decay,
        classifier_free_guidance,
    ) = [
        cfg['train'][task][_]
        for _ in (
            'use_amp',
            'num_workers',
            'num_epochs',
            'val_interval',
            'batch_size',
            'sw_batch_size',
            'lr',
            'effective_batch_size',
            'ema_decay',
            'classifier_free_guidance',
        )
    ]

    print('List Batch Size:\t', batch_size)
    print('Effective Batch:\t', effective_batch_size)

    patch_size = list(cfg['train']['vae']['patch_size'])

    val_prls, test_prls = set(cfg['val'].keys()), set(cfg['test'].keys())
    train_files, val_files, test_files = [], [], []

    for image_file in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(image_file.name.removesuffix('.npy').split('_')[:2])
        if prl in cfg['pairs']['excluded']:
            continue

        pid, rl = prl.split('_')
        f = dataset_root / 'pair' / pid / rl / 'context.toml'
        if f.exists():
            it = {'image': image_file.as_posix(), 'prl': prl, 'context': tomlkit.loads(f.read_text('utf-8')).unwrap()}
        else:
            raise RuntimeError(f'Non-exist {f.as_posix()}')

        if prl in test_prls:
            test_files.append(it)
        elif prl in val_prls:
            val_files.append(it)
        else:
            train_files.append(it)

    train_files.sort(key=lambda x: x['prl'])
    val_files.sort(key=lambda x: x['prl'])
    test_files.sort(key=lambda x: x['prl'])

    val_prl = val_files[0]['prl'] if len(val_files) else None

    print('Train:\t', len(train_files))
    print('Val:\t', len(val_files))

    def load_vae(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()

        print(f'[{subtask}]\t', f'Loading {ckpt_path}')

        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)

        print('Epoch:\t', loaded['epoch'])
        print('Channels:\t', channels := loaded['channels'])
        print('L1:   \t', loaded['val_l1'], 'best', loaded['best_val_l1'])
        print('PSNR:\t', loaded['val_psnr'])
        print('SSIM:\t', loaded['val_ssim'])
        print('Scale Factor:\t', sf := loaded['scale_factor'])
        print('Global Mean:\t', mean := loaded['global_mean'])

        vae = define.vae_kl(channels).to(device)
        vae.load_state_dict(loaded['state_dict'])
        vae.eval().float()
        print('Param:\t {0:.2f} B'.format(sum(p.numel() for p in vae.parameters()) / 1e9))

        i_val, r_val = 0.0, 0.0
        for metric in ('FID', 'Eikonal'):
            kw = f'i{metric.lower()}'
            if kw in loaded:
                print(f'i{metric}:\t', i_val := loaded[kw])
            kw = f'r{metric.lower()}'
            if kw in loaded:
                print(f'r{metric}:\t', r_val := loaded[kw])
        print('Interp/Recon:\t', i_val / (r_val + 1e-12))

        return vae, sf, mean

    vae_cond, cond_sf, cond_mean = load_vae('pre')
    vae_image, image_sf, image_mean = load_vae('metal')

    transforms = Compose(
        define.rflow_transforms(
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        )
    )

    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    # 动态 Volume 采样器
    # 最小 Shape: [32, 32, 44] (体积: 50,688)
    # 最大 Shape: [64, 88, 160] (体积最大达: 454,784)
    # 平均 Shape: [39, 37, 121] (平均体积: 181,923)
    # 中位数 Shape: [40, 36, 128] (中位数体积: 181,440)
    reference_volume = 180000
    max_volume = batch_size * reference_volume
    batch_sampler = define.DynamicRandomVolumeSampler(train_ds, max_volume=max_volume, shuffle=True)

    # 训练 Loader 使用 pad_list_data_collate 实现动态 Padding
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=pad_list_data_collate,
    )
    # 验证 Loader 保持 BS=1 即可
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    embed_dim = 256
    rflow = define.rflow_unet(context_embedding_size=embed_dim).to(device)
    context_embedder = define.ContextEmbedder(embed_dim=embed_dim).to(device)
    rflow_ema = define.EMA(rflow, decay=ema_decay)
    context_ema = define.EMA(context_embedder, decay=ema_decay)

    scheduler = define.scheduler_rflow()

    optimizer = torch.optim.AdamW(list(rflow.parameters()) + list(context_embedder.parameters()), lr=lr, weight_decay=1e-5)

    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    rflow_ckpt_path = (ckpt_dir / f'{task}_last.pt').resolve()

    if args.resume and rflow_ckpt_path.exists():
        try:
            print('Resuming:\t', rflow_ckpt_path)
            ckpt = torch.load(rflow_ckpt_path, map_location=device)
            rflow.load_state_dict(ckpt['rflow_state'])

            if 'context_state' in ckpt:
                print('Loading ContextEmbedder...')
                context_embedder.load_state_dict(ckpt['context_state'])

            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = lr

            if 'rflow_state_ema' in ckpt:
                rflow_ema.load_state_dict(ckpt['rflow_state_ema'])

            if 'context_state_ema' in ckpt:
                context_ema.load_state_dict(ckpt['context_state_ema'])

            if use_amp and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            start_epoch = ckpt['epoch']
            val_loss = ckpt.get('val_loss', float('inf'))

            # Explicitly delete the loaded checkpoint to free up system/GPU memory
            del ckpt
            torch.cuda.empty_cache()

            print('Epoch:\t', start_epoch)
            print('MSE:\t', val_loss)
            start_epoch += 1
        except Exception as e:
            print(f'Load failed: {e}')

    suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
    if args.resume:
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
            with torch.autocast(device_type=device.type, enabled=False):
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

        saver(recon[0].cpu(), meta_data={'filename_or_obj': f'{name}.nii.gz'})
        return recon.cpu()

    amp_ctx = autocast(device.type) if use_amp else nullcontext()

    accumulated_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, num_epochs):
        rflow.train()
        context_embedder.train()
        epoch_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1

            # batch 经过 pad_list_data_collate，全部是 Tensor
            image = batch['image'].to(device, non_blocking=True)
            cond = batch['condition'].to(device, non_blocking=True)
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)
            brand_id = batch['brand_id'].to(device, non_blocking=True)
            size_id = batch['size_id'].to(device, non_blocking=True)
            numerics = batch['numerics'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)

            current_bs = image.shape[0]

            with amp_ctx:
                # CFG 策略: 互斥的分支，支持 Batched
                cfg_rand = torch.rand(current_bs, 1, device=device)

                # 1. 10% 概率全局全部丢弃
                global_drop_mask = (cfg_rand < 0.10).float()

                # Image Cond: 运用 Global Drop
                cond = cond * (1.0 - global_drop_mask.view(current_bs, 1, 1, 1, 1))

                # Context Cond: 独立随机丢弃 (每个 Token 5% 概率)
                ind_drop_mask = (torch.rand(current_bs, 6, device=device) < 0.05).float()

                # 联合 Mask: 如果 global drop 则为 0，否则应用独立 drop
                current_masks = masks * (1.0 - global_drop_mask) * (1.0 - ind_drop_mask)

                # 生成全局条件 Embeddings [B, 6, C]
                context = context_embedder(brand_id, size_id, numerics, current_masks)

                # 采样时间步
                timesteps = scheduler.sample_timesteps(image)

                # 生成噪声，并严格屏蔽 Padding 区域
                noise = torch.randn_like(image) * valid_mask

                # RFM 加噪过程
                noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                # 拼接输入 (Image + Pre-op Condition)
                input_tensor = torch.cat([noisy_image, cond], dim=1)

                # 预测速度 (Velocity), 注入 Context
                velocity_pred = rflow(x=input_tensor, timesteps=timesteps, context=context)

                # RFM 的目标是真实数据与噪声的差: target_velocity = image - noise
                target_velocity = image - noise

                # 计算这一批次的有效体素 MSE (完美规避由于 Padding 导致的纯噪声区域 Loss 占比过高的问题)
                # 保证每个样本对梯度的贡献完全平等，避免体积偏差
                raw_mse = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float(), reduction='none')
                mse_per_sample = (raw_mse * valid_mask).sum(dim=(1, 2, 3, 4)) / (raw_mse.shape[1] * valid_mask.sum(dim=(1, 2, 3, 4)) + 1e-8)
                raw_mse = mse_per_sample.mean()

                # 记录原始 MSE 用于监控和打印
                display_loss = raw_mse.item()

                loss = raw_mse

                # 动态梯度累积缩放 (根据当前真实 bs 与期望有效 bs 的比例缩放 loss)
                micro_loss = loss * (current_bs / effective_batch_size)

                if use_amp:
                    scaler.scale(micro_loss).backward()
                else:
                    micro_loss.backward()

            accumulated_samples += current_bs

            if accumulated_samples >= effective_batch_size:
                if use_amp:
                    scaler.unscale_(optimizer)

                # 修正动态 Batch 带来的梯度误差，使得最终步进的梯度严格等于 accumulated_samples 个样本的均值
                scale_factor = effective_batch_size / accumulated_samples
                for param in list(rflow.parameters()) + list(context_embedder.parameters()):
                    if param.grad is not None:
                        param.grad *= scale_factor

                torch.nn.utils.clip_grad_norm_(list(rflow.parameters()) + list(context_embedder.parameters()), 1.0)

                if use_amp:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    step_skipped = scale_before > scale_after
                else:
                    optimizer.step()
                    step_skipped = False

                if not step_skipped:
                    rflow_ema.update(rflow)
                    context_ema.update(context_embedder)

                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0

            epoch_loss += display_loss

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss', display_loss, global_step)

            pbar.set_postfix({'MSE': f'{display_loss:.4f}'})

        writer.add_scalar('train/epoch_loss', epoch_loss / step, epoch)

        # 验证与采样 (保持 BS=1，不需要改 collate_fn)
        if epoch % val_interval == 0:
            rflow.eval()
            context_embedder.eval()
            rflow_ema.store(rflow)
            rflow_ema.copy_to(rflow)
            context_ema.store(context_embedder)
            context_ema.copy_to(context_embedder)

            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for i, batch in enumerate(val_bar := tqdm(val_loader, desc='Val')):
                    image = batch['image'].to(device)
                    cond = batch['condition'].to(device)
                    valid_mask = batch['valid_mask'].to(device)

                    brand_id = batch['brand_id'].to(device)
                    size_id = batch['size_id'].to(device)
                    numerics = batch['numerics'].to(device)
                    masks = batch['masks'].to(device)

                    # 生成全局条件 Embeddings
                    context = context_embedder(brand_id, size_id, numerics, masks)

                    timesteps = scheduler.sample_timesteps(image)
                    noise = torch.randn_like(image) * valid_mask
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    with amp_ctx:
                        velocity_pred = rflow(input_tensor, timesteps, context=context)
                        target_velocity = image - noise

                        # 验证同样使用 Valid Mask 防止 Padding 偏差
                        loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float(), reduction='none')
                        loss_per_sample = (loss * valid_mask).sum(dim=(1, 2, 3, 4)) / (loss.shape[1] * valid_mask.sum(dim=(1, 2, 3, 4)) + 1e-8)
                        loss = loss_per_sample.mean()

                    val_loss += loss.item()
                    val_steps += 1

                    prl = batch['prl'][0]
                    if prl == val_prl:
                        name = f'{prl}_{i}'

                        scheduler.set_timesteps(num_inference_steps=50)
                        all_timesteps = scheduler.timesteps
                        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

                        generator = torch.Generator(device=device).manual_seed(42)
                        # 初始化噪声时严格应用 valid_mask，防止 padding 区域引入 OOD 噪声
                        generated = torch.randn(image.shape, device=device, generator=generator) * valid_mask

                        # Pre-compute constant inputs for generation loop to save overhead
                        uncond = torch.zeros_like(cond)
                        cond_input = torch.cat([cond, uncond])

                        # CFG for context: (Cond Context, Uncond Context)
                        uncond_context = context_embedder(brand_id, size_id, numerics, torch.zeros_like(masks))
                        context_input = torch.cat([context, uncond_context])

                        for t, next_t in zip(all_timesteps, all_next_timesteps):
                            val_bar.set_postfix({'RFlow': t.item()})

                            latent_input = torch.cat([generated] * 2)
                            model_input = torch.cat([latent_input, cond_input], dim=1)

                            with torch.no_grad(), amp_ctx:  # 使用 AMP 保护以减少显存占用并加速推理
                                t_input = t[None].to(device).repeat(2)
                                velocity_pred_batch = rflow(model_input, t_input, context=context_input)

                            velocity_cond, velocity_uncond = velocity_pred_batch.chunk(2)
                            velocity_pred = velocity_uncond + classifier_free_guidance * (velocity_cond - velocity_uncond)

                            with torch.no_grad():
                                generated, _ = scheduler.step(velocity_pred, t, generated, next_t)
                                generated = generated * valid_mask  # 步进后必须再次清零 Padding 区域

                        val_bar.set_postfix({})

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
                                drr = fast_drr(img * 0.5 + 0.5, axis, th=(0.0, 1.0), mode='mean')
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

            rflow_ema.restore(rflow)
            context_ema.restore(context_embedder)
            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print('Val Loss:\t', avg_val_loss)

            ckpt = {
                'epoch': epoch,
                'rflow_state': rflow.state_dict(),
                'rflow_state_ema': rflow_ema.state_dict(),
                'context_state': context_embedder.state_dict(),
                'context_state_ema': context_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(ckpt, ckpt_dir / f'{task}_last.pt')

        torch.cuda.empty_cache()

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
