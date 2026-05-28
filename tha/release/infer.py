import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np


FEMORAL = {
    '': [''],
    'DePuy Corail': [
        '',
        '8 (标准无领)',
        '9 (标准无领)',
        '10 (标准无领)',
        '11 (标准无领)',
        '12 (标准无领)',
        '13 (标准无领)',
        '14 (标准无领)',
        '15 (标准无领)',
        '16 (标准无领)',
        '18 (标准无领)',
        '20 (标准无领)',
        '8 (标准带领)',
        '9 (标准带领)',
        '10 (标准带领)',
        '11 (标准带领)',
        '12 (标准带领)',
        '13 (标准带领)',
        '14 (标准带领)',
        '15 (标准带领)',
        '16 (标准带领)',
        '18 (标准带领)',
        '20 (标准带领)',
        '9 (高偏心无领)',
        '10 (高偏心无领)',
        '11 (高偏心无领)',
        '12 (高偏心无领)',
        '13 (高偏心无领)',
        '14 (高偏心无领)',
        '15 (高偏心无领)',
        '16 (高偏心无领)',
        '9 (内翻带领)',
        '10 (内翻带领)',
        '11 (内翻带领)',
        '12 (内翻带领)',
        '13 (内翻带领)',
        '14 (内翻带领)',
        '15 (内翻带领)',
        '16 (内翻带领)',
        '18 (内翻带领)',
        '20 (内翻带领)',
        '6 (DDH)',
        '10 (翻修标准)',
        '11 (翻修标准)',
        '12 (翻修标准)',
        '13 (翻修标准)',
        '14 (翻修标准)',
        '15 (翻修标准)',
        '16 (翻修标准)',
        '18 (翻修标准)',
        '20 (翻修标准)',
        '10 (翻修高偏心)',
        '11 (翻修高偏心)',
        '12 (翻修高偏心)',
        '13 (翻修高偏心)',
        '14 (翻修高偏心)',
        '15 (翻修高偏心)',
        '16 (翻修高偏心)',
        '18 (翻修高偏心)',
        '20 (翻修高偏心)',
    ],
    'DePuy Tri-Lock': ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    'DePuy SUMMIT': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'DePuy S-ROM': ['', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
    'Stryker Accolade TMZF': [
        '',
        '0 (127°)',
        '1 (127°)',
        '2 (127°)',
        '2.5 (127°)',
        '3 (127°)',
        '3.5 (127°)',
        '4 (127°)',
        '4.5 (127°)',
        '5 (127°)',
        '5.5 (127°)',
        '6 (127°)',
        '7 (127°)',
        '8 (127°)',
        '0 (132°)',
        '1 (132°)',
        '2 (132°)',
        '2.5 (132°)',
        '3 (132°)',
        '3.5 (132°)',
        '4 (132°)',
        '4.5 (132°)',
        '5 (132°)',
        '5.5 (132°)',
        '6 (132°)',
        '7 (132°)',
        '8 (132°)',
    ],
    'Stryker Accolade II': [
        '',
        '0 (127°)',
        '1 (127°)',
        '2 (127°)',
        '3 (127°)',
        '4 (127°)',
        '5 (127°)',
        '6 (127°)',
        '7 (127°)',
        '8 (127°)',
        '9 (127°)',
        '10 (127°)',
        '11 (127°)',
        '0 (132°)',
        '1 (132°)',
        '2 (132°)',
        '3 (132°)',
        '4 (132°)',
        '5 (132°)',
        '6 (132°)',
        '7 (132°)',
        '8 (132°)',
        '9 (132°)',
        '10 (132°)',
        '11 (132°)',
    ],
    'Stryker Secur-Fit': [
        '',
        '6 (127°)',
        '7 (127°)',
        '8 (127°)',
        '9 (127°)',
        '10 (127°)',
        '11 (127°)',
        '12 (127°)',
        '13 (127°)',
        '4 (132°)',
        '5 (132°)',
        '6 (132°)',
        '7 (132°)',
        '8 (132°)',
        '9 (132°)',
        '10 (132°)',
        '11 (132°)',
        '12 (132°)',
        '13 (132°)',
        '14 (132°)',
    ],
    'Wright Profemur': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'Smith & Nephew Synergy': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'],
    'Smith & Nephew Anthology': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    'Smith & Nephew Plus-TS': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Zimmer M/L Taper': ['', '4', '5', '6', '7.5', '9', '10', '11', '12.5', '13.5', '15', '16.25', '17.5', '20', '22.5', 'ML'],
    'Zimmer CLS Spotorno': ['', '5', '6', '7', '8', '9', '10', '11.25', '12.5', '13.75', '15', '16.25'],
    'AK Medical ML-TP': ['', '1', '2', '2.5', '3', '3.5', '4', '5', '6'],
    'Waldemar Link LCU': [
        '',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
    ],
    'Keyi Bangen SQKA': ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Zimmer CPT': ['', '0', '1', '2', '3', '4', '5', 'Long Size 2', 'Long Size 3', 'Long Size 4'],
    'Zimmer Wagner SL': ['', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
    'Wagner Cone': [
        '',
        '13 (125°)',
        '14 (125°)',
        '15 (125°)',
        '16 (125°)',
        '17 (125°)',
        '18 (125°)',
        '19 (125°)',
        '20 (125°)',
        '21 (125°)',
        '22 (125°)',
        '13 (135°)',
        '14 (135°)',
        '15 (135°)',
        '16 (135°)',
        '17 (135°)',
        '18 (135°)',
        '19 (135°)',
        '20 (135°)',
        '21 (135°)',
        '22 (135°)',
    ],
}

BRANDS = sorted(list(FEMORAL.keys()))
SIZES = sorted(list({s for sizes in FEMORAL.values() for s in sizes}))


def _printf(*args):
    print(*args)


def parse_context(brands, sizes, stem_brand=None, stem_size=None, cup_outer=None, head_outer=None, head_offset=None, liner_offset=None):
    import torch

    brand_to_id = {b: i for i, b in enumerate(brands)}
    size_to_id = {s: i for i, s in enumerate(sizes)}

    brand_val = stem_brand if stem_brand is not None else ''
    size_val = stem_size if stem_size is not None else ''

    if size_val and not brand_val:
        raise ValueError('Cannot specify stem_size without specifying stem_brand.')

    if brand_val not in FEMORAL:
        raise ValueError(f"Unknown stem_brand: '{brand_val}'. Supported brands are: {list(FEMORAL.keys())}")

    if size_val not in FEMORAL[brand_val]:
        raise ValueError(f"Unknown stem_size: '{size_val}' for brand '{brand_val}'. Supported sizes are: {FEMORAL[brand_val]}")

    brand_id = brand_to_id.get(brand_val, 0)
    brand_mask = 1.0 if brand_val in brand_to_id else 0.0

    size_id = size_to_id.get(size_val, 0)
    size_mask = 1.0 if size_val in size_to_id else 0.0

    def min_max_scale(val, min_val, max_val):
        if val is None or val == '':
            return 0.0, 0.0
        return 2.0 * (float(val) - min_val) / (max_val - min_val) - 1.0, 1.0

    cup_outer_val, cup_outer_mask = min_max_scale(cup_outer, 38.0, 62.0)
    head_outer_val, head_outer_mask = min_max_scale(head_outer, 22.0, 44.0)
    head_offset_val, head_offset_mask = min_max_scale(head_offset, -5.0, 9.0)
    liner_offset_val, liner_offset_mask = min_max_scale(liner_offset, 0.0, 6.0)

    nums = [cup_outer_val, head_outer_val, head_offset_val, liner_offset_val]
    masks = [brand_mask, size_mask, cup_outer_mask, head_outer_mask, head_offset_mask, liner_offset_mask]

    return (
        torch.tensor([brand_id], dtype=torch.long),
        torch.tensor([size_id], dtype=torch.long),
        torch.tensor([nums], dtype=torch.float32),
        torch.tensor([masks], dtype=torch.float32),
    )


def diff_dmc(volume, origin, spacing, threshold):
    """
    完全复制自 kernel.py:diff_dmc
    volume: [D, H, W] torch.Tensor
    """
    import torch
    import trimesh
    from diso import DiffDMC

    # volume 传入时应确保是 GPU Tensor 且为 float32
    # 这里的 -threshold 是因为 DiffDMC 默认找正值区域
    vertices, indices = DiffDMC(dtype=torch.float32)(-volume, None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()

    # 这里的 volume.shape 是 (D, H, W)
    # vertices 对应索引 [z, y, x]
    vertices = vertices * spacing * (np.array(volume.shape) - 1) + origin
    return trimesh.Trimesh(vertices, indices)


def main(
    cond,
    save,
    stem_brand=None,
    stem_size=None,
    cup_outer=None,
    head_outer=None,
    head_offset=None,
    liner_offset=None,
    vae_pre=None,
    vae_metal=None,
    rflow=None,
    cpu=False,
    amp=True,
    sw=4,
    seed=None,
    cfg=3.0,
    ts=5,
    printf=_printf,
):
    # 参数后处理
    sw_batch_size = max(sw, 1)
    cfg_val = max(float(cfg), 0.0)

    import time

    start_total = time.time()

    # 延迟导入以加快命令行响应速度
    import torch

    if not cpu:
        torch.backends.cudnn.benchmark = True

    from monai.inferers import sliding_window_inference
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
    from monai.networks.schedulers import RFlowScheduler
    from monai.transforms import CenterSpatialCrop, Compose, DivisiblePadd, EnsureChannelFirstd, LoadImaged, SpatialPadd
    from torch import autocast

    class ContextEmbedder(torch.nn.Module):
        """将手术设计参数编码为全局条件向量序列 [B, 6, C]"""

        def __init__(self, brands, sizes, embed_dim=256):
            super().__init__()
            self.brand_emb = torch.nn.Embedding(len(brands), embed_dim)
            self.size_emb = torch.nn.Embedding(len(sizes), embed_dim)

            self.cup_outer_proj = torch.nn.Linear(1, embed_dim)
            self.head_outer_proj = torch.nn.Linear(1, embed_dim)
            self.head_offset_proj = torch.nn.Linear(1, embed_dim)
            self.liner_offset_proj = torch.nn.Linear(1, embed_dim)

        def forward(self, brand_id, size_id, numerics, masks=None):
            brand_embed = self.brand_emb(brand_id)  # [B, C]
            size_embed = self.size_emb(size_id)  # [B, C]

            cup_outer_embed = self.cup_outer_proj(numerics[:, 0:1])  # [B, C]
            head_outer_embed = self.head_outer_proj(numerics[:, 1:2])  # [B, C]
            head_offset_embed = self.head_offset_proj(numerics[:, 2:3])  # [B, C]
            liner_offset_embed = self.liner_offset_proj(numerics[:, 3:4])  # [B, C]

            out = torch.stack([brand_embed, size_embed, cup_outer_embed, head_outer_embed, head_offset_embed, liner_offset_embed], dim=1)

            if masks is not None:
                out = out * masks.unsqueeze(-1)

            return out

    # 设备选择
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    printf('Device:\t {0}'.format(device))

    # 加载 VAE 模型 (AutoencoderKL)
    vae_dual = []
    for subtask in ('metal', 'pre'):
        vae_path_arg = {'pre': vae_pre, 'metal': vae_metal}.get(subtask)
        if vae_path_arg is None:
            vae_path = Path(__file__).parent.parent.parent.parent.parent / 'public-ssd/train-tha/checkpoints' / f'vae_{subtask}_best.pt'
        else:
            vae_path = Path(vae_path_arg)

        if not vae_path.exists():
            raise SystemError(f'Not found:\t {vae_path.resolve()}')
        else:
            printf('VAE:\t [{0}] {1}'.format(subtask, vae_path.resolve()))

        # 加载模型
        loaded = torch.load(vae_path, map_location='cpu', weights_only=False)

        printf('Epoch:\t', loaded['epoch'])
        printf('Channels:\t', channels := loaded['channels'])
        printf('Scale Factor:\t', sf := loaded['scale_factor'])
        printf('Global Mean:\t', mean := loaded['global_mean'])

        # 初始化 VAE 网络结构
        vae = AutoencoderKL(
            spatial_dims=3,
            in_channels=channels,
            out_channels=channels,
            num_res_blocks=(2, 2, 2),
            channels=(32, 64, 128),
            attention_levels=(False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            latent_channels=4,
            norm_num_groups=32,
            use_checkpoint=True,
        ).to(device)

        # 加载权重
        vae.load_state_dict(loaded['state_dict'])
        vae.eval().float()
        printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in vae.parameters()) / 1e9))

        vae_dual += [vae, sf, mean]

    vae_image, vae_image_scale, vae_image_mean, vae_cond, vae_cond_scale, vae_cond_mean = vae_dual

    # 加载 RFlow 模型 (DiffusionModelUNet)
    if rflow is None:
        rflow_path = Path(__file__).parent.parent.parent.parent.parent / 'public-ssd/train-tha/checkpoints' / 'rflow_last.pt'
    else:
        rflow_path = Path(rflow)

    if not rflow_path.exists():
        raise SystemError(f'Not found:\t {rflow_path.resolve()}')
    else:
        printf('RFlow:\t {0}'.format(rflow_path.resolve()))

    rflow_model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=8,
        num_res_blocks=(2, 2, 2),
        channels=(96, 192, 384),
        attention_levels=(False, False, True),
        norm_num_groups=32,
        with_conditioning=True,
        transformer_num_layers=2,
        cross_attention_dim=256,
        use_flash_attention=not cpu,
    ).to(device)

    context_embedder = ContextEmbedder(brands=BRANDS, sizes=SIZES, embed_dim=256).to(device)

    loaded = torch.load(rflow_path, map_location=device)
    printf('Epoch:\t {0}'.format(loaded['epoch']))
    printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in rflow_model.parameters()) / 1e9))

    if 'rflow_state_ema' in loaded:
        rflow_model.load_state_dict(loaded['rflow_state_ema'])
        printf('Loaded RFlow EMA weights.')
    elif 'rflow_state' in loaded:
        rflow_model.load_state_dict(loaded['rflow_state'])
    elif 'ema_state' in loaded:
        rflow_model.load_state_dict(loaded['ema_state'])
        printf('Loaded EMA weights (legacy).')
    else:
        rflow_model.load_state_dict(loaded['state_dict'])

    if 'context_state_ema' in loaded:
        context_embedder.load_state_dict(loaded['context_state_ema'])
        printf('Loaded Context EMA weights.')
    elif 'context_state' in loaded:
        context_embedder.load_state_dict(loaded['context_state'])
    else:
        printf('Warning: No ContextEmbedder state found in checkpoint.')

    rflow_model.eval().float()
    context_embedder.eval().float()

    # 初始化 RFlow 采样器
    scheduler = RFlowScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(ts)
    all_timesteps = scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

    if ts == 1:
        printf('RFlow 1-step mode: CFG is baked into the model weights, skipping dual-pass.')
    else:
        printf('CFG:\t {0} {1}'.format(cfg_val, 'Uncond-only' if cfg_val == 0 else 'Cond-only' if cfg_val == 1 else 'Guided'))

    printf('Steps:\t {0} (RFlow Sampling)'.format(ts))

    # 准备条件图像路径
    cond_path = Path(cond)
    if not cond_path.exists():
        raise SystemError(f'Condition not found:\t {cond_path.resolve()}')

    b_id, s_id, nums, msks = parse_context(BRANDS, SIZES, stem_brand, stem_size, cup_outer, head_outer, head_offset, liner_offset)
    b_id = b_id.to(device)
    s_id = s_id.to(device)
    nums = nums.to(device)
    msks = msks.to(device)
    print(b_id, s_id, nums, msks)

    with torch.no_grad():
        context_emb = context_embedder(b_id, s_id, nums, msks)
        uncond_context_emb = context_embedder(b_id, s_id, nums, torch.zeros_like(msks))

    # 准备后处理工具与原始尺寸
    import itk

    itk_img = itk.imread(cond_path.as_posix())
    original_itk_size = list(itk.size(itk_img))
    printf(f'Input Size:\t {original_itk_size}')

    cropper = CenterSpatialCrop(roi_size=original_itk_size)

    # 加载并归一化条件图像 [B, C, H, W, D]
    # MONAI ITKReader loads in (C, X, Y, Z) order
    cond_transforms = Compose([
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image']),
        SpatialPadd(keys=['image'], spatial_size=(128, 128, 128), constant_values=-1.0),
        DivisiblePadd(keys=['image'], k=16, constant_values=-1.0),
    ])
    cond_data = cond_transforms({'image': cond_path.as_posix()})
    cond_raw = cond_data['image']
    cond_tensor = cond_raw.unsqueeze(0).to(device)

    save_dir = Path(save) / '_'.join([
        cond_path.with_suffix('').with_suffix('').name,
        'seed',
        str(seed) if seed else 'random',
        'cfg',
        str(cfg),
        'ts',
        str(ts),
    ])
    save_dir.mkdir(parents=True, exist_ok=True)

    def decode_and_save(latent_tensor):
        """解码 Latent 並返回 (Z, Y, X, C) 格式的 numpy 数组"""
        # 1. 反向缩放 Latent
        z = latent_tensor / vae_image_scale + vae_image_mean

        # 2. VAE 解码
        def decode_predictor(inputs: torch.Tensor) -> torch.Tensor:
            with autocast(device.type) if amp else nullcontext():
                vae_latent_ch = vae_image.latent_channels
                if inputs.shape[1] > vae_latent_ch:
                    recons = []
                    for i in range(0, inputs.shape[1], vae_latent_ch):
                        recons.append(vae_image.decode(inputs[:, i : i + vae_latent_ch]))
                    return torch.cat(recons, dim=1)
                return vae_image.decode(inputs)

        with torch.no_grad():
            recon = sliding_window_inference(
                inputs=z,
                roi_size=[128 // 4 for _ in range(3)],
                sw_batch_size=sw_batch_size,
                predictor=decode_predictor,
                overlap=0.25,
                mode='gaussian',
                device=device,
                sw_device=device,
                progress=False,
            )

        # 3. 裁剪与后处理
        decoded = recon[0].detach().cpu().float()
        decoded = cropper(decoded)
        # decoded shape: [C, X, Y, Z]

        # ITK expects (Z, Y, X, C) order for a Vector Image of size (X, Y, Z)
        sdf_numpy = decoded.permute(3, 2, 1, 0).cpu().numpy()

        return sdf_numpy

    # VAE 编码 (Encoding)
    def encode_predictor(z):
        with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
            return vae_cond.encode(z)[0]

    with torch.no_grad():
        cond_encoded = sliding_window_inference(
            inputs=cond_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=sw_batch_size,
            predictor=encode_predictor,
            overlap=0.25,
            mode='gaussian',
            device=device,
            progress=True,
        )

    # 缩放 latent 分布
    cond_encoded = (cond_encoded - vae_cond_mean) * vae_cond_scale
    printf('VAE encode:\t {0:.2f}s'.format(time.time() - start_total))

    # 初始化随机噪声 z_0
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Generated image latent should have 8 channels (2 output channels * 4 latent channels)
    gen_shape = list(cond_encoded.shape)
    gen_shape[1] = 8
    generated = torch.randn(gen_shape, device=device, generator=generator)

    printf('RFlow generating...')

    # Euler 直线积分
    start_gen = time.time()
    for t, next_t in zip(all_timesteps, all_next_timesteps):
        # 预测直线速度场 (Velocity)
        with torch.no_grad():
            if ts == 1:
                # 2-RF Mode: Single pass
                model_input = torch.cat([generated, cond_encoded], dim=1)
                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    velocity_pred = rflow_model(model_input, t[None].to(device), context=context_emb)
            else:
                # 1-RF Mode or Multi-step: CFG logic
                t_input = t[None].to(device).repeat(2)

                if cfg_val > 1.0:
                    latent_input = torch.cat([generated] * 2, dim=0)
                    cond_input = torch.cat([cond_encoded, torch.zeros_like(cond_encoded)], dim=0)
                    context_input = torch.cat([context_emb, uncond_context_emb], dim=0)
                    model_input = torch.cat([latent_input, cond_input], dim=1)

                    with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                        velocity_pred_batch = rflow_model(model_input, t_input, context=context_input)

                    velocity_cond, velocity_uncond = velocity_pred_batch.chunk(2)
                    velocity_pred = velocity_uncond + cfg_val * (velocity_cond - velocity_uncond)
                elif cfg_val == 1.0:
                    model_input = torch.cat([generated, cond_encoded], dim=1)
                    with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                        velocity_pred = rflow_model(model_input, t[None].to(device), context=context_emb)
                else:
                    model_input = torch.cat([generated, torch.zeros_like(cond_encoded)], dim=1)
                    with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                        velocity_pred = rflow_model(model_input, t[None].to(device), context=uncond_context_emb)

        # Euler 更新步 (x_next = x_t + dt * v_t)
        with torch.no_grad():
            generated, _ = scheduler.step(velocity_pred, t, generated, next_t)

    printf('RFlow generate:\t {0:.2f}s'.format(time.time() - start_gen))

    # 最终解码并融合
    start_dec = time.time()
    generated_np = decode_and_save(generated)

    # 提取等值面网格体 (STL)
    if device.type != 'cpu':
        origin_xyz = np.array(itk_img.GetOrigin())
        spacing_xyz = np.array(itk_img.GetSpacing())

        for i, name_suffix in enumerate(['_cup', '_stem']):
            stl_name = cond_path.name.replace('.nii.gz', f'{name_suffix}.stl')
            stl_path = save_dir / stl_name

            final_sdf_torch = torch.from_numpy(generated_np[..., i]).to(device)

            spacing_zyx = spacing_xyz[[2, 1, 0]]
            origin_zyx = origin_xyz[[2, 1, 0]]

            mesh = diff_dmc(final_sdf_torch, origin_zyx, spacing_zyx, threshold=0.0)

            mesh.vertices = mesh.vertices[:, [2, 1, 0]]
            mesh.export(stl_path.as_posix())

    # 保存最终的结果
    # 方案 A: 保存为 4D 图像
    save_metal = save_dir / cond_path.name.replace('.nii.gz', '_metal.nii.gz')
    itk_metal = itk.image_from_array(generated_np.transpose(3, 0, 1, 2), is_vector=False)
    itk_metal.SetSpacing(list(itk_img.GetSpacing()) + [1.0])
    itk_metal.SetOrigin(list(itk_img.GetOrigin()) + [0.0])
    itk.imwrite(itk_metal, save_metal.as_posix())

    # 方案 B: 保存为独立文件
    for i, name_suffix in enumerate(['_cup', '_stem']):
        chan_path = save_dir / cond_path.name.replace('.nii.gz', f'{name_suffix}.nii.gz')
        itk_chan = itk.image_from_array(generated_np[..., i])
        itk_chan.SetSpacing(itk_img.GetSpacing())
        itk_chan.SetOrigin(itk_img.GetOrigin())
        itk_chan.SetDirection(itk_img.GetDirection())
        itk.imwrite(itk_chan, chan_path.as_posix())

    printf('Post-process:\t {0:.2f}s'.format(time.time() - start_dec))
    printf('Total time:\t {0:.2f}s'.format(time.time() - start_total))


if __name__ == '__main__':
    b = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description='Rectified Flow 推理脚本')

    parser.add_argument('--cond', type=str, required=True, help='术前条件图像路径 (.nii.gz)')
    parser.add_argument('--save', type=str, required=True, help='生成结果保存目录')

    parser.add_argument('--stem-brand', type=str, default=None, help='股骨柄型号')
    parser.add_argument('--stem-size', type=str, default=None, help='股骨柄规格')
    parser.add_argument('--cup-outer', type=float, default=None, help='髋臼杯外径')
    parser.add_argument('--head-outer', type=float, default=None, help='股骨头外径')
    parser.add_argument('--head-offset', type=float, default=None, help='股骨头偏距')
    parser.add_argument('--liner-offset', type=float, default=None, help='内衬偏心距')

    parser.add_argument('--vae-pre', type=str, default=None, help='VAE模型路径')
    parser.add_argument('--vae-metal', type=str, default=None, help='VAE模型路径')
    parser.add_argument('--rflow', type=str, default=None, help='RFlow模型路径')

    parser.add_argument('--cpu', action='store_true', default=False, help='强制使用 CPU 推理')
    parser.add_argument('--amp', action=b, default=True, help='是否启用混合精度')
    parser.add_argument('--sw', type=int, default=4, help='滑动窗口推理时的并行 Batch Size')

    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--cfg', type=float, default=3.0, help='Guidance 权重')
    parser.add_argument('--ts', type=int, default=5, help='采样步数')

    args = parser.parse_args()

    try:
        main(
            cond=args.cond,
            save=args.save,
            stem_brand=args.stem_brand,
            stem_size=args.stem_size,
            cup_outer=args.cup_outer,
            head_outer=args.head_outer,
            head_offset=args.head_offset,
            liner_offset=args.liner_offset,
            vae_pre=args.vae_pre,
            vae_metal=args.vae_metal,
            rflow=args.rflow,
            cpu=args.cpu,
            amp=args.amp,
            sw=args.sw,
            seed=args.seed,
            cfg=args.cfg,
            ts=args.ts,
        )
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
