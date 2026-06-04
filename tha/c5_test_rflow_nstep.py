import argparse
from pathlib import Path

import define
import numpy as np
import tomlkit
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import SSIMMetric
from monai.transforms import Compose
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Path to RFlow checkpoint (default: rflow_last.pt)')
    parser.add_argument('--num_samples', type=int, default=0, help='Max samples to test (0 for all)')
    parser.add_argument('--guidance', type=float, help='Override classifier_free_guidance from config')
    parser.add_argument('--steps_n', type=int, default=1, help='N-steps to compare against 50-steps')
    args = parser.parse_args()

    # 1. Load Config
    config_path = Path(args.config)
    cfg = tomlkit.loads(config_path.read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    dataset_root = Path(cfg['dataset']['root'])
    ckpt_dir = train_root / 'checkpoints'

    task = 'rflow'
    cfg_rflow = cfg['train'][task]
    cfg_vae = cfg['train']['vae']
    patch_size = list(cfg_vae['patch_size'])
    sw_batch_size = cfg_vae['sw_batch_size']

    # 2. Setup Data
    test_prls = set(cfg['test'].keys())
    test_files = []
    for f in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(f.name.removesuffix('.npy').split('_')[:2])
        if prl in test_prls and prl not in cfg['pairs']['excluded']:
            pid, rl = prl.split('_')
            ctx_f = dataset_root / 'pair' / pid / rl / 'context.toml'
            if ctx_f.exists():
                test_files.append({
                    'image': f.as_posix(),
                    'prl': prl,
                    'context': tomlkit.loads(ctx_f.read_text('utf-8')).unwrap()
                })
            else:
                raise RuntimeError(f'Non-exist {ctx_f.as_posix()}')

    if args.num_samples > 0:
        test_files = test_files[: args.num_samples]

    print(f'Test Samples: {len(test_files)}')

    # 3. Load VAE and Params
    def load_vae(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        vae = define.vae_kl(loaded['channels']).to(device)
        vae.load_state_dict(loaded['state_dict'])
        vae.eval()
        return vae, loaded['scale_factor'], loaded['global_mean']

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

    test_ds = Dataset(data=test_files, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=cfg_rflow['num_workers'])

    # 4. Load RFlow Model
    embed_dim = 256
    rflow = define.rflow_unet(context_embedding_size=embed_dim).to(device)
    context_embedder = define.ContextEmbedder(embed_dim=embed_dim).to(device)
    
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ckpt_dir / f'{task}_last.pt')
    print(f'Loading RFlow: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    if 'rflow_state_ema' in ckpt:
        rflow.load_state_dict(ckpt['rflow_state_ema'])
        print('Loaded RFlow EMA weights.')
    else:
        rflow.load_state_dict(ckpt['rflow_state'])
        print('Loaded RFlow raw state_dict.')

    if 'context_state_ema' in ckpt:
        context_embedder.load_state_dict(ckpt['context_state_ema'])
        print('Loaded ContextEmbedder EMA weights.')
    elif 'context_state' in ckpt:
        context_embedder.load_state_dict(ckpt['context_state'])
        print('Loaded ContextEmbedder raw state_dict.')

    rflow.eval()
    context_embedder.eval()

    scheduler = define.scheduler_rflow()
    cfg_guidance = args.guidance if args.guidance is not None else cfg_rflow['classifier_free_guidance']
    print(f'Comparison: {args.steps_n}-Step vs 50-Step (Guidance={cfg_guidance})')

    # 5. Test Loop
    l_psnrs, l_cosines = [], []
    i_psnrs, i_ssims = [], []

    def decode_to_img(z):
        z_norm = (z / image_sf + image_mean).detach().to(device).float()

        def predictor(inputs: torch.Tensor) -> torch.Tensor:
            ch = vae_image.latent_channels
            if inputs.shape[1] > ch:
                return torch.cat([vae_image.decode(inputs[:, i : i + ch]) for i in range(0, inputs.shape[1], ch)], dim=1)
            return vae_image.decode(inputs)

        return sliding_window_inference(
            inputs=z_norm,
            roi_size=[p // define.vae_downsample for p in patch_size],
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=0.25,
            mode='gaussian',
            device=device,
            sw_device=device,
            progress=False,
        )

    amp_ctx = torch.autocast(device_type=device.type)
    ssim_calc = SSIMMetric(spatial_dims=3, data_range=2.0)

    print(f'Starting Full Evaluation ({args.steps_n} vs 50)...')
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            image = batch['image'].to(device)
            cond = batch['condition'].to(device)
            uncond = torch.zeros_like(cond)

            brand_id = batch['brand_id'].to(device)
            size_id = batch['size_id'].to(device)
            numerics = batch['numerics'].to(device)
            masks = batch['masks'].to(device)

            context = context_embedder(brand_id, size_id, numerics, masks)
            uncond_context = context_embedder(brand_id, size_id, numerics, torch.zeros_like(masks))
            context_input = torch.cat([context, uncond_context])

            # Same noise for both inferences
            generator = torch.Generator(device=device).manual_seed(42)
            z0 = torch.randn(image.shape, device=device, generator=generator)

            # --- N-Step Inference ---
            scheduler.set_timesteps(num_inference_steps=args.steps_n)
            timesteps_n = scheduler.timesteps.to(device)
            next_timesteps_n = torch.cat([timesteps_n[1:], torch.zeros(1, device=device)])

            gn = z0.clone()
            with amp_ctx:
                for t, next_t in zip(timesteps_n, next_timesteps_n):
                    model_input = torch.cat([torch.cat([gn] * 2), torch.cat([cond, uncond])], dim=1)
                    t_input = t[None].to(device).repeat(2)
                    v_batch = rflow(model_input, t_input, context=context_input)
                    v_c, v_u = v_batch.chunk(2)
                    v_final = v_u + cfg_guidance * (v_c - v_u)
                    gn, _ = scheduler.step(v_final, t, gn, next_t)

            # --- 50-Step Inference ---
            scheduler.set_timesteps(num_inference_steps=50)
            timesteps_50 = scheduler.timesteps.to(device)
            next_timesteps_50 = torch.cat([timesteps_50[1:], torch.zeros(1, device=device)])

            g50 = z0.clone()
            with amp_ctx:
                for t, next_t in zip(timesteps_50, next_timesteps_50):
                    model_input = torch.cat([torch.cat([g50] * 2), torch.cat([cond, uncond])], dim=1)
                    t_input = t[None].to(device).repeat(2)
                    v_batch = rflow(model_input, t_input, context=context_input)
                    v_c, v_u = v_batch.chunk(2)
                    v_final = v_u + cfg_guidance * (v_c - v_u)
                    g50, _ = scheduler.step(v_final, t, g50, next_t)

            # --- Latent Evaluation ---
            l_cosines.append(F.cosine_similarity((gn - z0).flatten(), (g50 - z0).flatten(), dim=0).item())
            l_psnrs.append(-10 * np.log10(F.mse_loss(gn, g50).item() + 1e-10))

            # --- Image Evaluation ---
            img_n = decode_to_img(gn)
            img_50 = decode_to_img(g50)

            # Image PSNR
            i_psnrs.append(-10 * np.log10(F.mse_loss(img_n, img_50).item() + 1e-10))
            # Image SSIM
            ssim_calc(y_pred=img_n, y=img_50)
            i_ssims.append(ssim_calc.aggregate().item())
            ssim_calc.reset()

    # 6. Report
    print('\n' + '=' * 45)
    print(f'FINAL REPORT: {args.steps_n}-Step vs 50-Step')
    print('=' * 45)
    print(f'Latent Cosine Sim:   {np.mean(l_cosines):.4f}')
    print(f'Latent PSNR:         {np.mean(l_psnrs):.2f} dB')
    print('-' * 45)
    print(f'Image PSNR:          {np.mean(i_psnrs):.2f} dB')
    print(f'Image SSIM:          {np.mean(i_ssims):.4f}')
    print('=' * 45)


if __name__ == '__main__':
    main()
