from contextlib import nullcontext
from pathlib import Path
from typing import cast

import itk
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import CenterSpatialCrop, Compose, DivisiblePadd, EnsureChannelFirstd, Lambdad, LoadImaged, SpatialPadd
from torch import autocast

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

CUP_OUTER = [38.0, 62.0, 44.0, 2.0]
HEAD_OUTER = [22.0, 44.0, 32.0, 2.0]
HEAD_OFFSET = [-5.0, 9.0, 0.0, 1.0]
LINER_OFFSET = [0.0, 6.0, 0.0, 1.0]


def bone_normalize(ct_value: float) -> float:
    if 150.0 <= ct_value < 650.0:
        value = -1.0 + (ct_value - 150.0) / 500.0 * 1.0
    elif 650.0 <= ct_value < 1150.0:
        value = 0.0 + (ct_value - 650.0) / 500.0 * 0.5
    elif 1150.0 <= ct_value < 3150.0:
        value = 0.5 + (ct_value - 1150.0) / 2000.0 * 0.5
    elif ct_value >= 3150.0:
        value = 1.0
    else:
        value = -1.0
    return value


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

    cup_outer_val, cup_outer_mask = min_max_scale(cup_outer, *CUP_OUTER[:2])
    head_outer_val, head_outer_mask = min_max_scale(head_outer, *HEAD_OUTER[:2])
    head_offset_val, head_offset_mask = min_max_scale(head_offset, *HEAD_OFFSET[:2])
    liner_offset_val, liner_offset_mask = min_max_scale(liner_offset, *LINER_OFFSET[:2])

    nums = [cup_outer_val, head_outer_val, head_offset_val, liner_offset_val]
    masks = [brand_mask, size_mask, cup_outer_mask, head_outer_mask, head_offset_mask, liner_offset_mask]

    return (
        torch.tensor([brand_id], dtype=torch.long),
        torch.tensor([size_id], dtype=torch.long),
        torch.tensor([nums], dtype=torch.float32),
        torch.tensor([masks], dtype=torch.float32),
    )


def diff_dmc(volume, origin, spacing, direction, threshold):
    """
    volume: [X, Y, Z] torch.Tensor
    """
    import torch
    import trimesh
    from diso import DiffDMC

    # volume 传入时应确保是 GPU Tensor 且为 float32
    # 这里的 -threshold 是因为 DiffDMC 默认找正值区域
    vertices, indices = DiffDMC(dtype=torch.float32)(-volume, None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()

    # volume.shape 是 (X, Y, Z)
    # vertices 对应索引 [x, y, z] 归一化在 [0, 1] 之间
    I = vertices * (np.array(volume.shape) - 1)

    direction = np.array(direction)[:3, :3]
    spacing = np.array(spacing)[:3]

    # 物理坐标 = origin + direction * (I * spacing)
    physical = (I * spacing) @ direction.T + origin
    return trimesh.Trimesh(physical, indices)


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


def i1_load_models(vae_pre_path=None, vae_metal_path=None, rflow_path=None, cpu=False, printf=_printf):
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    printf('Device:\t {0}'.format(device))

    # 加载 VAE 模型 (AutoencoderKL)
    vae_dual = []
    for subtask in ('pre', 'metal'):
        arg = {'pre': vae_pre_path, 'metal': vae_metal_path}.get(subtask)
        if arg is None:
            f = Path(__file__).parent / f'vae_{subtask}_best.pt'
        else:
            f = Path(arg)

        if not f.exists():
            raise SystemError(f'Not found:\t {f.resolve()}')
        else:
            printf('VAE:\t [{0}] {1}'.format(subtask, f.resolve()))

        # 加载模型
        loaded = torch.load(f, map_location='cpu', weights_only=False)

        printf('Epoch:\t', loaded['epoch'])
        printf('Channels:\t', channels := loaded['channels'])
        printf('Scale Factor:\t', loaded['scale_factor'])
        printf('Global Mean:\t', loaded['global_mean'])

        # 初始化 VAE 网络结构
        vae_model = AutoencoderKL(
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
        vae_model.load_state_dict(loaded['state_dict'])
        vae_model.eval().float()
        printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in vae_model.parameters()) / 1e9))

        vae_dual.append((vae_model, loaded['scale_factor'], loaded['global_mean'], loaded['channels']))

    # 加载 RFlow 模型 (DiffusionModelUNet)
    if rflow_path is None:
        rflow_path = Path(__file__).parent / 'rflow_last.pt'
    else:
        rflow_path = Path(rflow_path)

    if not rflow_path.exists():
        raise SystemError(f'Not found:\t {rflow_path.resolve()}')
    else:
        printf('RFlow:\t {0}'.format(rflow_path.resolve()))

    loaded = torch.load(rflow_path, map_location=device)
    printf('Epoch:\t {0}'.format(loaded['epoch']))

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

    printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in rflow_model.parameters()) / 1e9))

    context_embedder = ContextEmbedder(brands=BRANDS, sizes=SIZES, embed_dim=256).to(device)

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

    return *vae_dual, (rflow_model, context_embedder)


def i2_context_embed(
    context_embedder, stem_brand=None, stem_size=None, cup_outer=None, head_outer=None, head_offset=None, liner_offset=None, cpu=False
):
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    brand_to_id = {b: i for i, b in enumerate(BRANDS)}
    size_to_id = {s: i for i, s in enumerate(SIZES)}

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

    cup_outer_val, cup_outer_mask = min_max_scale(cup_outer, *CUP_OUTER[:2])
    head_outer_val, head_outer_mask = min_max_scale(head_outer, *HEAD_OUTER[:2])
    head_offset_val, head_offset_mask = min_max_scale(head_offset, *HEAD_OFFSET[:2])
    liner_offset_val, liner_offset_mask = min_max_scale(liner_offset, *LINER_OFFSET[:2])

    nums = [cup_outer_val, head_outer_val, head_offset_val, liner_offset_val]
    masks = [brand_mask, size_mask, cup_outer_mask, head_outer_mask, head_offset_mask, liner_offset_mask]

    b_id, s_id, nums, msks = (
        torch.tensor([brand_id], dtype=torch.long).to(device),
        torch.tensor([size_id], dtype=torch.long).to(device),
        torch.tensor([nums], dtype=torch.float32).to(device),
        torch.tensor([masks], dtype=torch.float32).to(device),
    )

    with torch.no_grad():
        context_emb = context_embedder(b_id, s_id, nums, msks)
        context_emb_uncond = context_embedder(b_id, s_id, nums, torch.zeros_like(msks))

    return context_emb, context_emb_uncond


def i3_pre_encode(pre_path, vae_model, scale_factor, mean, channels, sw_batch_size=4, sw_overlap=0.25, cpu=False, amp=True):
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_path = Path(pre_path)
    if not pre_path.exists():
        raise RuntimeError(f'Pre image not found:\t {pre_path.resolve()}')

    itk_img = itk.imread(pre_path.as_posix())
    pre_origin = list(itk.origin(itk_img))
    pre_spacing = list(itk.spacing(itk_img))
    pre_size = list(itk.size(itk_img))

    direction = itk.GetArrayFromMatrix(itk_img.GetDirection())

    # 加载并归一化条件图像 [B, C, H, W, D]
    # MONAI ITKReader loads in (C, X, Y, Z) order
    transforms = Compose([
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image'], channel_dim=-1 if channels > 1 else 'no_channel'),
        Lambdad(keys=['image'], func=lambda x: torch.clamp(x, min=-1.0, max=1.0)),
        SpatialPadd(keys=['image'], spatial_size=(128, 128, 128), constant_values=-1.0),
        DivisiblePadd(keys=['image'], k=16, constant_values=-1.0),
    ])

    data = cast(dict, transforms({'image': pre_path.as_posix()}))
    raw = data['image']
    tensor = raw.unsqueeze(0).to(device)

    # VAE 编码 (Encoding)
    def encode_predictor(z):
        with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
            return vae_model.encode(z)[0]

    with torch.no_grad():
        encoded = sliding_window_inference(
            inputs=tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=sw_batch_size,
            predictor=encode_predictor,
            overlap=sw_overlap,
            mode='gaussian',
            device=device,
            progress=False,
        )

    # 缩放 latent 分布
    encoded = (encoded - mean) * scale_factor

    return encoded, pre_origin, pre_spacing, pre_size, direction


def i4_rflow_sample(rflow_model, context_emb, uncond_context_emb, pre_encoded, seed=None, ts=5, cfg=1.0, cpu=False, amp=True):
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    gen_shape = list(pre_encoded.shape)
    gen_shape[1] = 8
    generated = torch.randn(gen_shape, device=device, generator=generator)

    scheduler = RFlowScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(ts)
    all_timesteps = scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

    for t, next_t in zip(all_timesteps, all_next_timesteps):
        # 预测直线速度场 (Velocity)
        with torch.no_grad():
            if cfg > 1.0:
                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    model_input_cond = torch.cat([generated, pre_encoded], dim=1)
                    velocity_cond = rflow_model(model_input_cond, t[None].to(device), context=context_emb)

                    model_input_uncond = torch.cat([generated, torch.zeros_like(pre_encoded)], dim=1)
                    velocity_uncond = rflow_model(model_input_uncond, t[None].to(device), context=uncond_context_emb)

                velocity_pred = velocity_uncond + cfg * (velocity_cond - velocity_uncond)
            elif cfg == 1.0:
                model_input = torch.cat([generated, pre_encoded], dim=1)
                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    velocity_pred = rflow_model(model_input, t[None].to(device), context=context_emb)
            else:
                model_input = torch.cat([generated, torch.zeros_like(pre_encoded)], dim=1)
                with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                    velocity_pred = rflow_model(model_input, t[None].to(device), context=uncond_context_emb)

        # Euler 更新步 (x_next = x_t + dt * v_t)
        with torch.no_grad():
            generated, _ = scheduler.step(velocity_pred, t, generated, next_t)  # type: ignore

    return generated


def i5_metal_decode(generated, roi_size, vae_model, scale_factor, mean, channels, sw_batch_size=4, sw_overlap=0.25, cpu=False, amp=True):
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cropper = CenterSpatialCrop(roi_size=roi_size)

    # 反向缩放
    z = generated / scale_factor + mean

    # VAE 解码
    def decode_predictor(inputs: torch.Tensor) -> torch.Tensor:
        with autocast(device.type) if amp else nullcontext():
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
            roi_size=[128 // 4 for _ in range(3)],
            sw_batch_size=sw_batch_size,
            predictor=decode_predictor,
            overlap=sw_overlap,
            mode='gaussian',
            device=device,
            sw_device=device,
            progress=False,
        )

    # 还原尺寸
    decoded = recon[0].detach().cpu().float()
    decoded = cropper(decoded)
    decoded_np = decoded.cpu().numpy()

    return np.ascontiguousarray(decoded_np[0]), np.ascontiguousarray(decoded_np[1])


def i6_export(savedir, cup, stem, pre_path, origin, spacing, direction, cpu=False):
    if cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda':
        for name, tsdf in [('cup', cup), ('stem', stem)]:
            tensor = torch.from_numpy(tsdf).to(device)
            mesh = diff_dmc(tensor, origin, spacing, direction, threshold=0.0)
            mesh.export(savedir / f'{name}.stl')

        image = itk.imread(pre_path.as_posix())
        image = itk.array_from_image(image).transpose(2, 1, 0)
        image = np.ascontiguousarray(image)

        tensor = torch.from_numpy(image).to(device)
        mesh = diff_dmc(tensor, origin, spacing, direction, threshold=bone_normalize(226))
        mesh.export(savedir / 'pre.stl')

    for name, tsdf in [('cup', cup), ('stem', stem)]:
        tsdf_zyx = np.ascontiguousarray(tsdf.transpose(2, 1, 0))
        image = itk.image_from_array(tsdf_zyx)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(itk.GetMatrixFromArray(np.array(direction)[:3, :3]))
        itk.imwrite(image, savedir / f'{name}.nii.gz')

    cup_mask, stem_mask = cup > 0.0, stem > 0.0
    seg = (cup_mask | stem_mask).astype(np.uint8)
    seg_zyx = np.ascontiguousarray(seg.transpose(2, 1, 0))
    seg_img = itk.image_from_array(seg_zyx)
    seg_img.SetOrigin(origin)
    seg_img.SetSpacing(spacing)
    seg_img.SetDirection(itk.GetMatrixFromArray(np.array(direction)[:3, :3]))
    itk.imwrite(seg_img, savedir / 'metal.nii.gz')


if __name__ == '__main__':
    import tomlkit

    cfg = Path('config.toml')
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()

    prl = list(cfg['test'].keys())[1]
    pre_path = Path(cfg['train']['root']) / 'dataset' / 'pre' / f'{prl}.nii.gz'

    vae_pre, vae_metal, (rflow, context_embedder) = i1_load_models()
    context_emb, context_emb_uncond = i2_context_embed(context_embedder)
    pre_encoded, pre_origin, pre_spacing, pre_size, direction = i3_pre_encode(pre_path, *vae_pre)
    metal_latent = i4_rflow_sample(rflow, context_emb, context_emb_uncond, pre_encoded)
    cup, stem = i5_metal_decode(metal_latent, pre_size, *vae_metal)
    i6_export('save_infer', cup, stem, pre_path, pre_origin, pre_spacing, direction)

    print(pre_size, cup.shape, stem.shape)
