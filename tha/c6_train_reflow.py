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
from monai.transforms import Compose, MapTransform, SaveImage
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LoadReflowLatentConditiond(MapTransform):
    """Reads .npy file containing [cond, z_0, z_1_gen] pre-scaled latents"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        data_npy = np.load(d['image'])
        if isinstance(data_npy, np.ndarray):
            data_tensor = torch.from_numpy(data_npy).float()
        else:
            data_tensor = data_npy.float()

        d['condition'] = data_tensor[0:4]
        d['noise'] = data_tensor[4:12]
        d['image'] = data_tensor[12:20]  # This is z_1_gen
        return d


def reflow_collate_fn(batch):
    return {
        'image': [item['image'] for item in batch],
        'condition': [item['condition'] for item in batch],
        'noise': [item['noise'] for item in batch],
        'prl': [item.get('prl', '') for item in batch],
    }


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = tomlkit.loads(config_path.read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'reflow'
    (
        use_amp,
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
        cfg['train']['rflow'][_]
        for _ in (
            'use_amp',
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

    lr *= 0.5  # Reflow 属于微调阶段，通常需要更小的学习率

    gradient_accumulation_steps = max(1, gradient_accumulation_steps // batch_size)
    print('List Batch Size:\t', batch_size)
    print('Grad Accu Steps:\t', gradient_accumulation_steps)

    patch_size = list(cfg['train']['vae']['patch_size'])

    val_prls = set(cfg['val'].keys())
    train_files, val_files = [], []

    # Load Reflow training data
    for f in (train_root / 'latents_reflow').glob('*.npy'):
        prl = '_'.join(f.name.removesuffix('.npy').split('_')[:2])
        train_files.append({'image': f.as_posix(), 'prl': prl})

    # Load Validation data from original latents
    for f in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(f.name.removesuffix('.npy').split('_')[:2])
        if prl in val_prls:
            val_files.append({'image': f.as_posix(), 'prl': prl})

    val_prl = val_files[0]['prl'] if len(val_files) else None

    print('Train (Reflow):\t', len(train_files))
    print('Val (Original):\t', len(val_files))

    def load_vae(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        sf = loaded['scale_factor']
        mean = loaded['global_mean']
        vae = define.vae_kl(loaded['channels']).to(device)
        vae.load_state_dict(loaded['state_dict'])
        vae.eval().float()
        return vae, sf, mean

    vae_cond, cond_sf, cond_mean = load_vae('pre')
    vae_image, image_sf, image_mean = load_vae('metal')

    # Train transforms (already scaled by c5a_prepare_reflow_data.py)
    train_transforms = Compose([LoadReflowLatentConditiond(keys=['image'])])

    # Val transforms (needs scaling on the fly)
    val_transforms = Compose(
        define.rflow_transforms(
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        )
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=reflow_collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    rflow = define.rflow_unet().to(device)
    ema = define.EMA(rflow, decay=ema_decay)

    reflow_ckpt_path = (ckpt_dir / f'{task}_last.pt').resolve()
    resume_exists = args.resume and reflow_ckpt_path.exists()

    # Initialize from the pre-trained 1-RF model ONLY if NOT resuming from a 2-RF checkpoint
    if not resume_exists:
        pretrained_path = (ckpt_dir / 'rflow_last.pt').resolve()
        if pretrained_path.exists():
            print('Initializing Reflow model from 1-RF:\t', pretrained_path)
            ckpt = torch.load(pretrained_path, map_location=device)
            print('Epoch:\t', ckpt['epoch'])

            rflow.load_state_dict(ckpt['state_dict'])
            if 'ema_state' in ckpt:
                ema.load_state_dict(ckpt['ema_state'])

            # Explicitly delete the loaded checkpoint to free up system/GPU memory
            del ckpt
            torch.cuda.empty_cache()

    scheduler = define.scheduler_rflow()

    optimizer = torch.optim.AdamW(rflow.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0

    if resume_exists:
        try:
            print('Resuming Reflow:\t', reflow_ckpt_path)
            ckpt = torch.load(reflow_ckpt_path, map_location=device)
            rflow.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if 'ema_state' in ckpt:
                ema.load_state_dict(ckpt['ema_state'])
            if use_amp and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch']

            # Explicitly delete the loaded checkpoint to free up system/GPU memory
            del ckpt
            torch.cuda.empty_cache()

            print('Epoch:\t', start_epoch)
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

    for epoch in range(start_epoch, num_epochs):
        rflow.train()
        epoch_loss = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1

            image_list = batch['image']
            cond_list = batch['condition']
            noise_list = batch['noise']

            current_bs = len(image_list)
            display_loss = 0.0

            with amp_ctx:
                for b_idx in range(current_bs):
                    image = image_list[b_idx].unsqueeze(0).to(device, non_blocking=True)
                    cond = cond_list[b_idx].unsqueeze(0).to(device, non_blocking=True)
                    noise = noise_list[b_idx].unsqueeze(0).to(device, non_blocking=True)

                    drop_mask = (torch.rand(1, 1, 1, 1, 1, device=device) < 0.15).float()
                    cond = cond * (1.0 - drop_mask)

                    timesteps = scheduler.sample_timesteps(image)

                    # For Reflow, x_t = t * x_1 + (1-t) * x_0
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)

                    input_tensor = torch.cat([noisy_image, cond], dim=1)
                    velocity_pred = rflow(x=input_tensor, timesteps=timesteps)

                    # Target velocity is exactly image (z_1_gen) - noise (z_0)
                    target_velocity = image - noise

                    loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float())

                    # Micro-batching: 计算完单张图的 loss 立刻 backward 释放计算图
                    micro_loss = loss / current_bs / gradient_accumulation_steps

                    if use_amp:
                        scaler.scale(micro_loss).backward()
                    else:
                        micro_loss.backward()

                    display_loss += micro_loss.item() * gradient_accumulation_steps

            if use_amp:
                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(rflow.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(rflow)
            else:
                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(rflow.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(rflow)

            epoch_loss += display_loss

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss', display_loss, global_step)

            pbar.set_postfix({'MSE': f'{display_loss:.4f}'})

        writer.add_scalar('train/epoch_loss', epoch_loss / step, epoch)

        if epoch % val_interval == 0:
            rflow.eval()
            ema.store(rflow)
            ema.copy_to(rflow)

            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for i, batch in enumerate(val_bar := tqdm(val_loader, desc='Val')):
                    image = batch['image'].to(device)
                    cond = batch['condition'].to(device)

                    # Compute val loss
                    timesteps = scheduler.sample_timesteps(image)
                    generator = torch.Generator(device=device).manual_seed(42)
                    noise = torch.randn(image.shape, device=device, generator=generator)
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    with amp_ctx:
                        velocity_pred = rflow(input_tensor, timesteps)
                        target_velocity = image - noise
                        loss = torch.nn.functional.mse_loss(velocity_pred.float(), target_velocity.float())

                    val_loss += loss.item()
                    val_steps += 1

                    prl = batch['prl'][0]
                    if prl == val_prl:
                        name = f'{prl}_{i}'

                        # REFLOW MAGIC: Set inference steps to 1 !
                        num_inference_steps = 1
                        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
                        all_timesteps = scheduler.timesteps
                        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

                        generator = torch.Generator(device=device).manual_seed(42)
                        generated = torch.randn(image.shape, device=device, generator=generator)

                        for t, next_t in zip(all_timesteps, all_next_timesteps):
                            val_bar.set_postfix({'1-Step Inference': t.item()})

                            if num_inference_steps == 1:
                                # 2-RF Mode: Single pass (CFG baked into the model weights)
                                model_input = torch.cat([generated, cond], dim=1)
                                with torch.no_grad():
                                    velocity_pred = rflow(model_input, t[None].to(device))
                            else:
                                # 1-RF Mode: Dual pass CFG
                                latent_input = torch.cat([generated] * 2)
                                uncond = torch.zeros_like(cond)
                                cond_input = torch.cat([cond, uncond])
                                model_input = torch.cat([latent_input, cond_input], dim=1)

                                with torch.no_grad():
                                    t_input = t[None].to(device).repeat(2)
                                    velocity_pred_batch = rflow(model_input, t_input)

                                velocity_cond, velocity_uncond = velocity_pred_batch.chunk(2)
                                velocity_pred = velocity_uncond + classifier_free_guidence * (velocity_cond - velocity_uncond)

                            with torch.no_grad():
                                generated, _ = scheduler.step(velocity_pred, t, generated, next_t)

                        with amp_ctx:
                            vis_generated = decode(generated, f'{name}_val_epoch_{epoch:03d}_Gen', vae_image, image_sf, image_mean, epoch)
                            vis_gt = decode(image, f'{name}_val_epoch_{epoch:03d}_GT', vae_image, image_sf, image_mean, epoch)
                            vis_cond = decode(cond, f'{name}_val_epoch_{epoch:03d}_Cond', vae_cond, cond_sf, cond_mean, epoch)

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

            ema.restore(rflow)
            avg_val_loss = val_loss / val_steps
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            print('Val Loss:\t', avg_val_loss)

            ckpt = {
                'epoch': epoch,
                'state_dict': rflow.state_dict(),
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
