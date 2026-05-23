import argparse
import hashlib
from pathlib import Path

import define
import numpy as np
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
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
    ckpt_dir = train_root / 'checkpoints'
    out_dir = train_root / 'latents_reflow'
    out_dir.mkdir(parents=True, exist_ok=True)

    test_prls = set(cfg['test'].keys())
    val_prls = set(cfg['val'].keys())
    train_files = []

    for f in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(f.name.removesuffix('.npy').split('_')[:2])
        if prl in cfg['pairs']['excluded'] or prl in test_prls or prl in val_prls:
            continue
        train_files.append({'image': f.as_posix(), 'prl': prl})

    print('Files to process for Reflow:\t', len(train_files))

    def load_vae_stats(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()
        loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return loaded['scale_factor'], loaded['global_mean']

    cond_sf, cond_mean = load_vae_stats('pre')
    image_sf, image_mean = load_vae_stats('metal')

    # Load original unscaled latents and scale them
    transforms = Compose(
        define.rflow_transforms(
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        )
    )

    ds = Dataset(data=train_files, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    lfm = define.rflow_unet().to(device)
    lfm_ckpt_path = (ckpt_dir / 'lfm_last.pt').resolve()
    print('Loading LFM from:\t', lfm_ckpt_path)

    ckpt = torch.load(lfm_ckpt_path, map_location=device)
    print('Epoch:\t', ckpt['epoch'])

    if 'ema_state' in ckpt:
        print('Using EMA weights for generation.')
        for name, param in lfm.named_parameters():
            if name in ckpt['ema_state']:
                param.data.copy_(ckpt['ema_state'][name])
    else:
        lfm.load_state_dict(ckpt['state_dict'])
    lfm.eval()

    scheduler = define.scheduler_rflow()
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    all_timesteps = scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

    cfg_scale = cfg['train']['lfm']['classifier_free_guidance']

    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating Reflow Trajectories'):
            cond = batch['condition'].to(device)
            image = batch['image'].to(device)
            prl = batch['prl'][0]

            # Use deterministic seed per sample to ensure reproducible z_0
            seed = int(hashlib.sha256(prl.encode('utf-8')).hexdigest(), 16) % (2**32)
            generator = torch.Generator(device=device).manual_seed(seed)

            z_0 = torch.randn_like(image, device=device, generator=generator)
            generated = z_0.clone()

            for t, next_t in zip(all_timesteps, all_next_timesteps):
                latent_input = torch.cat([generated] * 2)
                uncond = torch.zeros_like(cond)
                cond_input = torch.cat([cond, uncond])
                model_input = torch.cat([latent_input, cond_input], dim=1)

                t_input = t[None].to(device).repeat(2)
                with torch.amp.autocast(device.type):
                    velocity_pred_batch = lfm(model_input, t_input)

                velocity_cond, velocity_uncond = velocity_pred_batch.chunk(2)
                velocity_pred = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)

                generated, _ = scheduler.step(velocity_pred, t, generated, next_t)

            # Save pre-scaled tensors [cond, z_0, z_1_gen]
            cond_cpu = cond[0].cpu().numpy()
            z_0_cpu = z_0[0].cpu().numpy()
            z_1_gen_cpu = generated[0].cpu().numpy()

            # Shape will be [12, D, H, W]
            out_tensor = np.concatenate([cond_cpu, z_0_cpu, z_1_gen_cpu], axis=0)
            np.save(out_dir / f'{prl}.npy', out_tensor)

    print('Reflow data generation completed.')


if __name__ == '__main__':
    main()
