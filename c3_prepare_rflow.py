import argparse
from contextlib import nullcontext
from pathlib import Path

import define
import numpy as np
import tomlkit
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from torch.amp import autocast
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--sw_batch_size', type=int, default=36)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    train_root = Path(cfg['train']['root'])
    dataset_root = train_root / 'dataset'
    ckpt_dir = train_root / 'checkpoints'

    latents_root = train_root / 'latents'

    task = 'vae'
    patch_size = cfg['train'][task]['patch_size']
    use_amp = cfg['train'][task]['use_amp']

    def load_vae(subtask):
        load_pt = ckpt_dir / f'{task}_{subtask}_best.pt'
        print(f'VAE {subtask}')
        print(f'Loading checkpoint from {load_pt}...')
        try:
            checkpoint = torch.load(load_pt, map_location=device, weights_only=False)
            channels = int(checkpoint['channels'])
            print(f'Channels: {channels}')

            vae_model = define.vae_kl(channels).to(device)
            vae_model.load_state_dict(checkpoint['state_dict'])

            if 'scale_factor' in checkpoint:
                scale_factor = checkpoint['scale_factor']
                global_mean = checkpoint.get('global_mean', 0.0)
                print(f'Scale Factor ({subtask}): {scale_factor:.6f}, Mean: {global_mean:.6f}')
            else:
                raise SystemExit(f'Scale factor not prepared for {subtask}')

            vae_model.eval()
            return vae_model, channels
        except Exception as e:
            raise SystemExit(f'Failed to load {subtask} checkpoint: {e}')

    vae_pre, ch_pre = load_vae('pre')
    vae_metal, ch_metal = load_vae('metal')

    transforms_pre = Compose(define.vae_val_transforms(patch_size, ch_pre))
    transforms_metal = Compose(define.vae_val_transforms(patch_size, ch_metal))

    def encode_predictor(model):
        def _predictor(inputs: torch.Tensor) -> torch.Tensor:
            return model.encode(inputs)[0]

        return _predictor

    pre_dir = dataset_root / 'pre'
    metal_dir = dataset_root / 'metal'

    latents_root.mkdir(parents=True, exist_ok=True)
    pre_files = sorted(list(pre_dir.glob('*.nii.gz')))

    for pre_path in tqdm(pre_files):
        prl = '_'.join(pre_path.name.removesuffix('.nii.gz').split('_')[:2])
        if prl in cfg['pairs']['excluded']:
            continue

        cup_path = metal_dir / f'{prl}_cup.nii.gz'
        stem_path = metal_dir / f'{prl}_stem.nii.gz'
        save_path = latents_root / f'{prl}.npy'

        z_channels = []
        for path, vae, transforms in [
            (pre_path, vae_pre, transforms_pre),
            (cup_path, vae_metal, transforms_metal),
            (stem_path, vae_metal, transforms_metal),
        ]:
            batch = transforms({'image': path.as_posix()})['image']

            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch)

            images = batch.unsqueeze(0).to(device)

            with torch.no_grad():
                amp_ctx = autocast(device.type) if use_amp else nullcontext()
                with amp_ctx:
                    z = sliding_window_inference(
                        inputs=images,
                        roi_size=patch_size,
                        sw_batch_size=args.sw_batch_size,
                        predictor=encode_predictor(vae),
                        overlap=0.25,
                        mode='gaussian',
                        device=device,
                        sw_device=device,
                        progress=False,
                    )

            # 不应用 scale factor，保持原始 latent 存盘
            z = z.cpu().numpy()
            z_channels.append(z)

        z_pre, z_cup, z_stem = z_channels

        # 通道拼接，[0:4] 条件 pre, [4:8] 目标 cup [8:12] 目标 stem
        z_cat = np.concatenate((z_pre[0], z_cup[0], z_stem[0]), axis=0).astype(np.float32)
        np.save(save_path, z_cat)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
