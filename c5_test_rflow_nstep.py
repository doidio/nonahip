import argparse
import json
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
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Path to RFlow checkpoint (default: rflow_last.pt)')
    parser.add_argument('--num_samples', type=int, default=0, help='Max samples to test (0 for all)')
    parser.add_argument('--steps_n', type=int, default=10, help='N-steps to compare against 50-steps')
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
                    'ctx_raw': tomlkit.dumps(tomlkit.loads(ctx_f.read_text('utf-8')).unwrap()),
                    'condition': (train_root / 'cond' / f.name).as_posix()
                })
            else:
                raise RuntimeError(f'Non-exist {ctx_f.as_posix()}')

    if args.num_samples > 0:
        test_files = test_files[: args.num_samples]

    print(f'Test Samples: {len(test_files)}')

    # 3. Load VAE and Transforms
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
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=cfg_rflow.get('num_workers', 4))

    # 4. Load Models (RFlow, ContextEmbedder, ParamHead, PubMedBERT)
    print('Loading PubMedBERT...')
    tokenizer = AutoTokenizer.from_pretrained(cfg_rflow['text_encoder_path'])
    text_encoder = AutoModel.from_pretrained(cfg_rflow['text_encoder_path']).to(device)
    text_encoder.eval()

    print('Loading Generation Models...')
    rflow = define.rflow_unet(context_embedding_size=768).to(device)
    context_embedder = define.ContextEmbedder().to(device)
    param_head = define.ParameterVelocityHead().to(device)
    
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ckpt_dir / f'{task}_last.pt')
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    def load_weight(model, name):
        if f'{name}_ema' in ckpt:
            model.load_state_dict(ckpt[f'{name}_ema'])
            print(f'Loaded {name} EMA weights.')
        elif name in ckpt:
            model.load_state_dict(ckpt[name])
            print(f'Loaded {name} raw state_dict.')
            
    load_weight(rflow, 'rflow_state')
    load_weight(context_embedder, 'context_state')
    load_weight(param_head, 'param_state')

    rflow.eval()
    context_embedder.eval()
    param_head.eval()

    scheduler = define.scheduler_rflow()
    print(f'Comparison: {args.steps_n}-Step vs 50-Step (Joint Evolution)')

    # 5. Test Loop
    l_psnrs, l_cosines = [], []
    i_psnrs, i_ssims = [], []
    c_cosines_N_vs_50, c_cosines_N_vs_True, c_cosines_50_vs_True = [], [], []

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

            # Get Ground Truth Text Embedding
            c_text_strs = []
            for b in range(image.shape[0]):
                ctx_str = batch['ctx_raw'][b] if 'ctx_raw' in batch else '{}'
                ctx = json.loads(ctx_str)
                c_text_strs.append(define.generate_text(ctx, level='full'))

            tokens = tokenizer(c_text_strs, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            outputs = text_encoder(**tokens)
            attn_mask = tokens['attention_mask']
            token_embs = outputs.last_hidden_state
            input_mask = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
            c_text_true = torch.sum(token_embs * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

            # Same initial noise for both inferences
            generator = torch.Generator(device=device).manual_seed(42)
            z0_y = torch.randn(image.shape, device=device, generator=generator)
            z0_c = torch.randn(c_text_true.shape, device=device, generator=generator)

            # --- N-Step Inference ---
            scheduler.set_timesteps(num_inference_steps=args.steps_n)
            timesteps_n = scheduler.timesteps.to(device)
            next_timesteps_n = torch.cat([timesteps_n[1:], torch.zeros(1, dtype=timesteps_n.dtype, device=device)])

            gn_y = z0_y.clone()
            gn_c = z0_c.clone()
            with amp_ctx:
                for t, next_t in zip(timesteps_n, next_timesteps_n):
                    t_input = t[None].to(device)
                    current_context = context_embedder(gn_c)
                    model_input = torch.cat([gn_y, cond], dim=1)
                    
                    v_y = rflow(model_input, t_input, context=current_context)
                    v_c = param_head(model_input, t_input)
                    
                    gn_y, _ = scheduler.step(v_y, t, gn_y, next_t)
                    gn_c, _ = scheduler.step(v_c, t, gn_c, next_t)

            # --- 50-Step Inference ---
            scheduler.set_timesteps(num_inference_steps=50)
            timesteps_50 = scheduler.timesteps.to(device)
            next_timesteps_50 = torch.cat([timesteps_50[1:], torch.zeros(1, dtype=timesteps_50.dtype, device=device)])

            g50_y = z0_y.clone()
            g50_c = z0_c.clone()
            with amp_ctx:
                for t, next_t in zip(timesteps_50, next_timesteps_50):
                    t_input = t[None].to(device)
                    current_context = context_embedder(g50_c)
                    model_input = torch.cat([g50_y, cond], dim=1)
                    
                    v_y = rflow(model_input, t_input, context=current_context)
                    v_c = param_head(model_input, t_input)
                    
                    g50_y, _ = scheduler.step(v_y, t, g50_y, next_t)
                    g50_c, _ = scheduler.step(v_c, t, g50_c, next_t)

            # --- Evaluation ---
            l_cosines.append(F.cosine_similarity((gn_y - z0_y).flatten(), (g50_y - z0_y).flatten(), dim=0).item())
            l_psnrs.append(-10 * np.log10(F.mse_loss(gn_y, g50_y).item() + 1e-10))

            c_cosines_N_vs_50.append(F.cosine_similarity(gn_c, g50_c, dim=1).mean().item())
            c_cosines_N_vs_True.append(F.cosine_similarity(gn_c, c_text_true, dim=1).mean().item())
            c_cosines_50_vs_True.append(F.cosine_similarity(g50_c, c_text_true, dim=1).mean().item())

            img_n = decode_to_img(gn_y)
            img_50 = decode_to_img(g50_y)

            i_psnrs.append(-10 * np.log10(F.mse_loss(img_n, img_50).item() + 1e-10))
            ssim_calc(y_pred=img_n, y=img_50)
            i_ssims.append(ssim_calc.aggregate().item())
            ssim_calc.reset()

    # 6. Report
    print('\n' + '=' * 60)
    print(f'FINAL REPORT: {args.steps_n}-Step vs 50-Step (Joint Evolution)')
    print('=' * 60)
    print(f'Geometry (y) Latent PSNR:   {np.mean(l_psnrs):.2f} dB')
    print(f'Geometry (y) Latent CosSim: {np.mean(l_cosines):.4f}')
    print(f'Geometry (y) Image PSNR:    {np.mean(i_psnrs):.2f} dB')
    print(f'Geometry (y) Image SSIM:    {np.mean(i_ssims):.4f}')
    print('-' * 60)
    print(f'Parameter (c) CosSim [N vs 50]:   {np.mean(c_cosines_N_vs_50):.4f}')
    print(f'Parameter (c) CosSim [N vs True]: {np.mean(c_cosines_N_vs_True):.4f}')
    print(f'Parameter (c) CosSim [50 vs True]:{np.mean(c_cosines_50_vs_True):.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
