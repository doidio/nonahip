import argparse
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SaveImage
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import define
from kernel import fast_drr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # 妥协与性能优化折中说明：
    # 禁用 cudnn.benchmark 是为了避免动态尺寸输入时，cuDNN 频繁检测并重新编译 3D 卷积计算图带来的开销。
    # 因为每个样本的 ROI 尺寸不一致，逐个读取训练时输入尺度在不断变化。
    # 禁用后虽然失去了针对固定尺寸的极致优化，但能彻底根除由于尺寸切换产生的每步卡顿与重编译延迟。
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = tomlkit.loads(config_path.read_text('utf-8')).unwrap()
    progress_enabled = sys.stderr.isatty()

    task = 'rflow'
    resume = bool(args.resume or cfg['train'][task].get('resume', False))

    train_root = Path(str(cfg['train']['root']))
    dataset_root = Path(cfg['dataset']['root'])
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    (
        use_amp,
        num_workers,
        num_epochs,
        val_interval,
        sw_batch_size,
        lr,
        effective_batch_size,
        ema_decay,
    ) = [
        cfg['train'][task][_]
        for _ in (
            'use_amp',
            'num_workers',
            'num_epochs',
            'val_interval',
            'sw_batch_size',
            'lr',
            'effective_batch_size',
            'ema_decay',
        )
    ]
    use_amp = bool(use_amp and device.type == 'cuda')

    print('Effective Batch:\t', effective_batch_size)

    patch_size = list(cfg['train']['vae']['patch_size'])

    val_prls, test_prls = set(cfg['val'].keys()), set(cfg['test'].keys())
    train_files, val_files = [], []

    for image_file in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(image_file.name.removesuffix('.npy').split('_')[:2])
        if prl in cfg['pairs']['excluded'] or prl in test_prls:
            continue

        pid, rl = prl.split('_')
        f = dataset_root / 'pair' / pid / rl / 'context.toml'
        if f.exists():
            it = {'image': image_file.as_posix(), 'prl': prl, 'context': tomlkit.loads(f.read_text('utf-8')).unwrap()}
        else:
            raise RuntimeError(f'Non-exist {f.as_posix()}')

        if prl in val_prls:
            val_files.append(it)
        else:
            train_files.append(it)

    train_files.sort(key=lambda x: x['prl'])
    val_files.sort(key=lambda x: x['prl'])

    print('Train:\t', len(train_files))
    print('Val:\t', len(val_files))
    if not train_files:
        raise RuntimeError('Empty training split.')
    if not val_files:
        raise RuntimeError('Empty validation split.')

    val_prl = val_files[0]['prl']

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
    pin_memory = device.type == 'cuda'
    persistent_workers = num_workers > 0

    # 妥协与机制调整说明：
    # 为了避免批次间的动态 Padding (导致背景噪声区占比较大以及引入人为边界伪影)，
    # 我们将 batch_size 设为 1，逐个加载单样本进行训练。
    # 由于每个样本单独加载，不再存在多样本拼 Batch 时的尺寸对齐需求，故完全取消了动态 Padding 整理函数 (collate_fn)
    # 和动态体积采样器 (batch_sampler)。
    # 显存及优化稳定性方面，通过梯度累加在固定样本数后执行一次优化器更新
    # 依然能够实现宏观上的大 Batch 均值效应，确保训练的稳定性与收敛效果。
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    # 验证 Loader 保持 BS=1 即可
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    embed_dim = 256
    rflow = define.rflow_unet(context_embedding_size=embed_dim).to(device)
    condition_encoder = define.StructuredProsthesisConditionEncoder(embed_dim=embed_dim).to(device)
    rflow_ema = define.EMA(rflow, decay=ema_decay)
    condition_ema = define.EMA(condition_encoder, decay=ema_decay)

    scheduler = define.scheduler_rflow()
    condition_modes = define.PROSTHESIS_CONDITION_MODES
    condition_weight_cfg = cfg['train'][task].get('condition_mode_weights', {})
    condition_mode_weights = tuple(float(condition_weight_cfg.get(mode, 1.0)) for mode in condition_modes)
    if any(weight < 0 for weight in condition_mode_weights):
        raise ValueError('Condition mode weights must be non-negative.')
    if sum(condition_mode_weights) <= 0:
        raise ValueError('At least one condition mode weight must be positive.')
    condition_mode_labels = define.PROSTHESIS_CONDITION_MODE_LABELS
    print('Condition Mode Weights:\t', {condition_mode_labels[mode]: weight for mode, weight in zip(condition_modes, condition_mode_weights)})

    trainable_params = list(rflow.parameters()) + list(condition_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)

    scaler = GradScaler() if use_amp else None

    start_epoch = 0

    # 继续训练
    if resume:
        load_pt = (ckpt_dir / f'{task}_last.pt').resolve()
    else:
        load_pt = None

    if load_pt and load_pt.exists():
        try:
            print('Resuming:\t', load_pt)
            ckpt = torch.load(load_pt, map_location=device)
            rflow.load_state_dict(ckpt['rflow_state'])
            print('Loading StructuredProsthesisConditionEncoder...')
            condition_encoder.load_state_dict(ckpt['condition_encoder_state'])

            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = lr

            if 'rflow_state_ema' in ckpt:
                rflow_ema.load_state_dict(ckpt['rflow_state_ema'])

            condition_ema.load_state_dict(ckpt['condition_encoder_state_ema'])

            if use_amp and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            start_epoch = ckpt['epoch']
            val_geometry_loss = ckpt['val_geometry_velocity_mse']

            # Explicitly delete the loaded checkpoint to free up system/GPU memory
            del ckpt
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            print('Epoch:\t', start_epoch)
            print('Val Geometry Velocity MSE:\t', val_geometry_loss)
            start_epoch += 1
        except Exception as e:
            raise SystemError(f'Load failed: {e}')

    # 日志
    if resume:
        candidates = []
        if log_dir.exists():
            for p in log_dir.iterdir():
                if p.is_dir() and p.name.startswith(f'{task}_'):
                    candidates.append(p)
        if candidates:
            # 根据目录名中的时间戳排序 (如 rflow_20260609_160842)
            prefix = f'{task}_'

            def get_sort_key(p):
                name = p.name
                if name.startswith(prefix):
                    ts = name[len(prefix) : len(prefix) + 15]
                    parts = ts.split('_')
                    if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit() and len(parts[1]) == 6:
                        return (ts, name)
                return ('', p.name)

            candidates.sort(key=get_sort_key)
            log_dir = candidates[-1]
            print('Resuming logs in:\t', log_dir)
        else:
            suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S_resume')
            log_dir = log_dir / suffix
            print('No existing log directory found for resume. Creating:\t', log_dir)
    else:
        suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
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

    def decode(z, name, vae_model, sf, mean):
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

    def amp_context():
        return autocast(device_type=device.type, enabled=use_amp) if use_amp else nullcontext()

    accumulated_samples = 0
    optimizer.zero_grad(set_to_none=True)

    def optimizer_step(accumulated_count):
        if accumulated_count <= 0:
            return

        if use_amp:
            scaler.unscale_(optimizer)

        # Losses are scaled by effective_batch_size during accumulation; rescale
        # the tail step so a partial final micro-batch still becomes a true mean.
        tail_scale = effective_batch_size / accumulated_count
        for param in trainable_params:
            if param.grad is not None:
                param.grad *= tail_scale

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

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
            condition_ema.update(condition_encoder)

        optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, num_epochs):
        rflow.train()
        condition_encoder.train()
        epoch_geometry_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}', disable=not progress_enabled)

        for batch in pbar:
            step += 1

            image = batch['image'].to(device, non_blocking=True)
            bone_latent = batch['condition'].to(device, non_blocking=True)
            stem_model_id = batch['stem_model_id'].to(device, non_blocking=True)
            stem_spec_id = batch['stem_spec_id'].to(device, non_blocking=True)
            numerics = batch['numerics'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)
            current_bs = image.shape[0]

            current_bone = bone_latent.clone()
            current_masks = masks.clone()
            for i in range(current_bs):
                mode = random.choices(condition_modes, weights=condition_mode_weights, k=1)[0]
                current_bone[i : i + 1], current_masks[i : i + 1] = define.apply_prosthesis_condition_mode(
                    bone_latent[i : i + 1],
                    masks[i : i + 1],
                    mode,
                )

            with amp_context():
                # 采样时间步
                timesteps = scheduler.sample_timesteps(image)

                # 训练数据已离线对齐，且 batch_size = 1；此处不再需要运行时 padding mask。
                noise = torch.randn_like(image)

                # RFM 加噪过程
                noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                # 结构化假体条件 tokens [B, 6, C]
                condition_tokens = condition_encoder(stem_model_id, stem_spec_id, numerics, current_masks)

                # 拼接加噪假体潜变量与术前骨骼潜变量
                input_tensor = torch.cat([noisy_image, current_bone], dim=1)

                # 预测假体潜变量速度，并通过交叉注意力注入结构化条件
                velocity_pred = rflow(x=input_tensor, timesteps=timesteps, context=condition_tokens)

                target_velocity = image - noise

                geometry_loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float(), reduction='mean')
                disp_geometry_loss = geometry_loss.item()

                # 动态梯度累积缩放 (根据当前真实 bs 与期望有效 bs 的比例缩放 loss)
                micro_loss = geometry_loss * (current_bs / effective_batch_size)

                if use_amp:
                    scaler.scale(micro_loss).backward()
                else:
                    micro_loss.backward()

            accumulated_samples += current_bs

            if accumulated_samples >= effective_batch_size:
                optimizer_step(accumulated_samples)
                accumulated_samples = 0

            epoch_geometry_loss += disp_geometry_loss

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('loss/geometry_velocity_mse_train_step', disp_geometry_loss, global_step)

            if progress_enabled:
                pbar.set_postfix({'geom_mse': f'{disp_geometry_loss:.4f}'})

        # Flush the final partial accumulation before validation/checkpointing.
        if accumulated_samples > 0:
            optimizer_step(accumulated_samples)
            accumulated_samples = 0

        writer.add_scalar('loss/geometry_velocity_mse_train_epoch', epoch_geometry_loss / step, epoch)

        # 验证与采样 (保持 BS=1，不需要改 collate_fn)
        if epoch % val_interval == 0:
            rflow.eval()
            condition_encoder.eval()
            rflow_ema.store(rflow)
            rflow_ema.copy_to(rflow)
            condition_ema.store(condition_encoder)
            condition_ema.copy_to(condition_encoder)

            val_geometry_loss_sums = {mode: 0.0 for mode in condition_modes}
            val_steps = 0

            with torch.no_grad():
                for val_idx, batch in enumerate(val_bar := tqdm(val_loader, desc='Val', disable=not progress_enabled)):
                    image = batch['image'].to(device)
                    bone_latent = batch['condition'].to(device)
                    stem_model_id = batch['stem_model_id'].to(device)
                    stem_spec_id = batch['stem_spec_id'].to(device)
                    numerics = batch['numerics'].to(device)
                    masks = batch['masks'].to(device)

                    timesteps = scheduler.sample_timesteps(image)

                    # 验证同样按单样本动态尺寸执行，不引入运行时 padding mask。
                    noise = torch.randn_like(image)
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
                    target_velocity = image - noise

                    for mode in condition_modes:
                        current_bone, current_masks = define.apply_prosthesis_condition_mode(bone_latent, masks, mode)
                        condition_tokens = condition_encoder(stem_model_id, stem_spec_id, numerics, current_masks)
                        input_tensor = torch.cat([noisy_image, current_bone], dim=1)

                        with amp_context():
                            velocity_pred = rflow(input_tensor, timesteps, context=condition_tokens)
                            geometry_loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float(), reduction='mean')

                        val_geometry_loss_sums[mode] += geometry_loss.item()
                    val_steps += 1

                    prl = batch['prl'][0]
                    if prl == val_prl:
                        name = f'{prl}_{val_idx}'
                        current_bone, current_masks = define.apply_prosthesis_condition_mode(
                            bone_latent,
                            masks,
                            'bone_prosthesis_full',
                        )
                        condition_tokens = condition_encoder(stem_model_id, stem_spec_id, numerics, current_masks)

                        scheduler.set_timesteps(num_inference_steps=50)
                        all_timesteps = scheduler.timesteps
                        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

                        generator = torch.Generator(device=device).manual_seed(42)
                        generated = torch.randn(image.shape, device=device, generator=generator)

                        for t, next_t in zip(all_timesteps, all_next_timesteps):
                            if progress_enabled:
                                val_bar.set_postfix({'RFlow': t.item()})

                            with torch.no_grad(), amp_context():
                                t_input = t[None].to(device)

                                model_input = torch.cat([generated, current_bone], dim=1)

                                velocity_pred = rflow(model_input, t_input, context=condition_tokens)

                            with torch.no_grad():
                                generated, _ = scheduler.step(velocity_pred, t, generated, next_t)

                        if progress_enabled:
                            val_bar.set_postfix({})

                        with amp_context():
                            vis_generated = decode(generated, f'{name}_val_epoch_{epoch:03d}_GeneratedImplant', vae_image, image_sf, image_mean)
                            vis_gt = decode(image, f'{name}_val_epoch_{epoch:03d}_TargetImplant', vae_image, image_sf, image_mean)
                            vis_bone = decode(current_bone, f'{name}_val_epoch_{epoch:03d}_BoneCondition', vae_cond, cond_sf, cond_mean)

                        # DRR Visualization (Refer to VAE style)
                        axis = 1
                        val_vis_dir = log_dir / 'val'
                        val_vis_dir.mkdir(parents=True, exist_ok=True)

                        def get_drr_hstack(vis_tensor):
                            drrs = []
                            for c in range(vis_tensor.shape[1]):
                                img = vis_tensor[0, c].numpy()
                                drr = fast_drr(img + 1.0, axis, th=(0.1, 2.0), mode='mean')
                                drrs.append(np.flipud(drr.transpose(1, 0, 2)))
                            return np.hstack(drrs)

                        drr_gen = get_drr_hstack(vis_generated)
                        drr_gt = get_drr_hstack(vis_gt)
                        drr_bone = get_drr_hstack(vis_bone)

                        writer.add_image(f'val_sample/{name}/generated_implant', drr_gen, epoch, dataformats='HWC')
                        writer.add_image(f'val_sample/{name}/target_implant', drr_gt, epoch, dataformats='HWC')
                        writer.add_image(f'val_sample/{name}/bone_condition', drr_bone, epoch, dataformats='HWC')

                        Image.fromarray(drr_gen).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_GeneratedImplant.png')
                        Image.fromarray(drr_gt).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_TargetImplant.png')
                        Image.fromarray(drr_bone).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_BoneCondition.png')

                        # Diff DRR (hstack)
                        diff_drrs = []
                        for c in range(vis_generated.shape[1]):
                            diff = np.abs(vis_generated[0, c].numpy() - vis_gt[0, c].numpy())
                            drr_diff = fast_drr(diff + 1.0, axis, th=(0.1, 2.0), mode='mean')
                            diff_drrs.append(np.flipud(drr_diff.transpose(1, 0, 2)))

                        drr_diff_hstack = np.hstack(diff_drrs)
                        writer.add_image(f'val_sample/{name}/absolute_error', drr_diff_hstack, epoch, dataformats='HWC')
                        Image.fromarray(drr_diff_hstack).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_AbsoluteError.png')

            rflow_ema.restore(rflow)
            condition_ema.restore(condition_encoder)
            val_geometry_loss_by_mode = {mode: val_geometry_loss_sums[mode] / val_steps for mode in condition_modes}
            val_geometry_loss = val_geometry_loss_by_mode['bone_prosthesis_full']
            writer.add_scalar('loss/geometry_velocity_mse_val_full_condition', val_geometry_loss, epoch)
            for mode, loss_value in val_geometry_loss_by_mode.items():
                writer.add_scalar(f'loss_by_condition/{mode}', loss_value, epoch)

            val_loss_text = ' | '.join(f'{condition_mode_labels[mode]}: {value:.4f}' for mode, value in val_geometry_loss_by_mode.items())
            print(f'Val Geometry MSE: {val_geometry_loss:11.4f} | {val_loss_text}')

            ckpt = {
                'epoch': epoch,
                'rflow_state': rflow.state_dict(),
                'rflow_state_ema': rflow_ema.state_dict(),
                'condition_encoder_state': condition_encoder.state_dict(),
                'condition_encoder_state_ema': condition_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_geometry_velocity_mse': val_geometry_loss,
                'val_geometry_velocity_mse_by_mode': val_geometry_loss_by_mode,
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(ckpt, ckpt_dir / f'{task}_last.pt')

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
