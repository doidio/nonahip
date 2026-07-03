from copy import deepcopy

import numpy as np
import torch
from b0_preload_prothesis import FEMORAL
from monai.losses import PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import (
    CopyItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    MapTransform,
    RandCropByPosNegLabeld,
    SpatialPadd,
)

ct_min = -1024.0
ct_bone_min = 150.0  # 用于归一化
ct_bone_best = 220.0  # 用于配准和显示
ct_metal = 2500.0

# TotalSegmentator 标签
ct_seg_femur_left = 75
ct_seg_femur_right = 76
ct_seg_hip_left = 77
ct_seg_hip_right = 78

roi_spacing = 1.0  # 重采样体素精度 mm
sdf_t = 5.0  # 截断距离 mm


vae_downsample = 4

FEMORAL_STEM_MODELS = tuple(sorted(FEMORAL.keys()))
FEMORAL_STEM_SPECS = tuple((model, size) for model in FEMORAL_STEM_MODELS for size in FEMORAL[model])
PROSTHESIS_NUMERIC_RANGES = {
    'cup_outer': (38.0, 62.0),
    'head_outer': (20.0, 44.0),
    'head_offset': (-6.0, 9.0),
    'liner_offset': (0.0, 6.0),
}
PROSTHESIS_CONDITION_MODES = (
    'unconditional',
    'bone_only',
    'prosthesis_full_only',
    'bone_stem_model',
    'bone_stem_model_spec',
    'bone_prosthesis_full',
)
PROSTHESIS_CONDITION_MODE_LABELS = {
    'unconditional': 'Unconditional',
    'bone_only': 'Bone only',
    'prosthesis_full_only': 'Prosthesis parameters only',
    'bone_stem_model': 'Bone + stem model',
    'bone_stem_model_spec': 'Bone + stem model/spec',
    'bone_prosthesis_full': 'Bone + full prosthesis parameters',
}
STEM_MODEL_TOKEN = 0
STEM_MODEL_SPEC_TOKEN = 1


def enforce_prosthesis_condition_dependencies(prosthesis_masks):
    squeeze = prosthesis_masks.ndim == 1
    if squeeze:
        prosthesis_masks = prosthesis_masks.unsqueeze(0)
    prosthesis_masks = prosthesis_masks.clone()
    prosthesis_masks[:, STEM_MODEL_SPEC_TOKEN] *= prosthesis_masks[:, STEM_MODEL_TOKEN]
    return prosthesis_masks.squeeze(0) if squeeze else prosthesis_masks


def apply_prosthesis_condition_mode(bone_latent, prosthesis_masks, mode):
    bone_latent = bone_latent.clone()
    prosthesis_masks = enforce_prosthesis_condition_dependencies(prosthesis_masks)

    if mode == 'unconditional':
        bone_latent.zero_()
        prosthesis_masks.zero_()
    elif mode == 'bone_only':
        prosthesis_masks.zero_()
    elif mode == 'prosthesis_full_only':
        bone_latent.zero_()
    elif mode == 'bone_stem_model':
        prosthesis_masks[:, STEM_MODEL_SPEC_TOKEN:] = 0.0
    elif mode == 'bone_stem_model_spec':
        prosthesis_masks[:, STEM_MODEL_SPEC_TOKEN + 1 :] = 0.0
    elif mode == 'bone_prosthesis_full':
        pass
    else:
        raise ValueError(f'Unknown prosthesis condition mode: {mode}')

    return bone_latent, enforce_prosthesis_condition_dependencies(prosthesis_masks)


def vae_kl(channels: int):
    return AutoencoderKL(
        spatial_dims=3,
        in_channels=channels,
        out_channels=channels,
        num_res_blocks=(2, 2, 2),
        channels=(32, 64, 128),  # 逐层加宽，捕捉高频骨纹理
        attention_levels=(
            False,
            False,
            False,
        ),  # 自编码器必须采用纯卷积，Patch Training 与 Attention 之间天然矛盾
        with_encoder_nonlocal_attn=False,  # 关闭非局部注意力
        with_decoder_nonlocal_attn=False,  # 关闭非局部注意力
        latent_channels=4,  # 保持 4 通道，足够编码密度信息
        norm_num_groups=32,  # 归一化层，也会削弱 Patch Training 效果
        use_checkpoint=True,
    )


def vae_discriminator(channels: int):
    return PatchDiscriminator(
        spatial_dims=3,
        channels=64,  # 起始通道数
        in_channels=channels,  # 输入与编码器一致
        out_channels=1,  # 输出必须是单通道 (Real/Fake Score)
        num_layers_d=3,  # 3层下采样，感受野适中，关注局部纹理细节
    )


def vae_perceptual_loss():
    return PerceptualLoss(
        spatial_dims=3,
        network_type='medicalnet_resnet50_23datasets',
        is_fake_3d=False,
        pretrained=True,
    )


def _foreground_fn(x):
    return (x > -0.95).float()


def _clamp_fn(x):
    return torch.clamp(x, min=-1.0, max=1.0)


def vae_train_transforms(patch_size, channels):
    # 设计妥协说明：
    # 离线数据生成已将 ROI 大小对齐填充为 32 的整数倍。
    # 当前 VAE (下采样4倍) 与 RFlow UNet (下采样4倍) 组合要求最小倍数为 16。
    # 此处使用固定对齐因子 32 可以为以后微调网络参数（如增加下采样深度）预留足够的兼容余量，
    # 避免由于网络架构调整而频繁重新生成庞大的离线训练数据。
    # 此外，因为数据源和 patch_size（128）已天生是 32 的倍数，此处不再需要运行时的 DivisiblePadd。
    return [
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image'], channel_dim=-1 if channels > 1 else 'no_channel'),
        Lambdad(keys=['image'], func=_clamp_fn),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=-1.0),
        CopyItemsd(keys=['image'], times=1, names=['label']),
        Lambdad(keys=['label'], func=_foreground_fn),
        RandCropByPosNegLabeld(
            keys=['image'],
            label_key='label',
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=1,
        ),
        DeleteItemsd(keys=['label']),
    ]


def vae_val_transforms(patch_size, channels):
    # 设计妥协说明同上。数据源已离线对齐 32 倍数，故运行时无需 DivisiblePadd 逻辑。
    return [
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image'], channel_dim=-1 if channels > 1 else 'no_channel'),
        Lambdad(keys=['image'], func=_clamp_fn),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=-1.0),
    ]


class LoadRFlowLatentsd(MapTransform):
    """读取 RFlow 训练用潜变量：[术前骨骼 4 通道, 假体 TSDF 8 通道]"""

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
        d['image'] = data_tensor[4:12]

        return d


class ScaleLatentd(MapTransform):
    """根据 VAE 统计值对 Latent 进行归一化"""

    def __init__(self, keys, image_mean, image_sf, cond_mean, cond_sf, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.image_mean = image_mean
        self.image_sf = image_sf
        self.cond_mean = cond_mean
        self.cond_sf = cond_sf

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key == 'image':
                d[key] = (d[key] - self.image_mean) * self.image_sf
            elif key == 'condition':
                d[key] = (d[key] - self.cond_mean) * self.cond_sf
        return d


def _has_context_value(x):
    if x is None:
        return False
    if isinstance(x, str):
        text = x.strip().lower()
        return text != '' and text not in {'nan', 'none', 'null'}
    if isinstance(x, (list, tuple)):
        return any(_has_context_value(item) for item in x)
    if isinstance(x, (int, float, np.number)):
        return bool(np.isfinite(x))
    return x != ''


def _strip_context_string(x):
    return x.strip() if isinstance(x, str) else x


def _get_context_value(ctx, key, prefer_best=True):
    if prefer_best:
        best_value = ctx.get(f'{key}_best')
        if _has_context_value(best_value):
            return best_value
    return ctx.get(key)


def _as_context_sequence(value, length):
    if isinstance(value, (list, tuple)):
        sequence = list(value)
    elif _has_context_value(value):
        sequence = [value]
    else:
        sequence = []
    return [sequence[i] if i < len(sequence) else '' for i in range(length)]


def _get_context_sequence(ctx, key, length, prefer_best=True):
    raw_values = _as_context_sequence(ctx.get(key), length)
    if not prefer_best:
        return raw_values
    best_values = _as_context_sequence(ctx.get(f'{key}_best'), length)
    return [best if _has_context_value(best) else raw for raw, best in zip(raw_values, best_values)]


def _scale_context_number(value, min_value, max_value):
    if not _has_context_value(value):
        return 0.0, 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not np.isfinite(value):
        return 0.0, 0.0
    scaled = 2.0 * (value - min_value) / (max_value - min_value) - 1.0
    return float(np.clip(scaled, -1.0, 1.0)), 1.0


class PrepareProsthesisConditiond(MapTransform):
    """提取结构化假体条件"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.stem_model_to_id = {model: i for i, model in enumerate(FEMORAL_STEM_MODELS)}
        self.stem_spec_to_id = {spec: i for i, spec in enumerate(FEMORAL_STEM_SPECS)}

    def _encode_category(self, value, mapping):
        value = _strip_context_string(value)
        if _has_context_value(value) and value in mapping:
            return mapping[value], 1.0
        return mapping.get('', 0), 0.0

    def _encode_stem_spec(self, femoral_model, femoral_size):
        femoral_model = _strip_context_string(femoral_model)
        femoral_size = _strip_context_string(femoral_size)
        if _has_context_value(femoral_model) and _has_context_value(femoral_size):
            if femoral_size in FEMORAL.get(femoral_model, []):
                return self.stem_spec_to_id[(femoral_model, femoral_size)], 1.0
        return self.stem_spec_to_id.get(('', ''), 0), 0.0

    def __call__(self, data):
        d = dict(data)
        ctx = d.get('context', {})

        femoral_spec = _get_context_sequence(ctx, 'femoral_spec', length=2)
        stem_model = femoral_spec[0] if len(femoral_spec) >= 1 else ''
        stem_size = femoral_spec[1] if len(femoral_spec) >= 2 else ''

        stem_model_id, stem_model_mask = self._encode_category(stem_model, self.stem_model_to_id)
        stem_spec_id, stem_spec_mask = self._encode_stem_spec(stem_model, stem_size)

        cup_outer = _get_context_value(ctx, 'cup_outer')
        head_outer = _get_context_value(ctx, 'head_outer')
        head_offset = _get_context_value(ctx, 'head_offset')
        liner_offset = ctx.get('liner_offset')

        cup_outer_value, cup_outer_mask = _scale_context_number(cup_outer, *PROSTHESIS_NUMERIC_RANGES['cup_outer'])
        head_outer_value, head_outer_mask = _scale_context_number(head_outer, *PROSTHESIS_NUMERIC_RANGES['head_outer'])
        head_offset_value, head_offset_mask = _scale_context_number(head_offset, *PROSTHESIS_NUMERIC_RANGES['head_offset'])
        liner_offset_value, liner_offset_mask = _scale_context_number(liner_offset, *PROSTHESIS_NUMERIC_RANGES['liner_offset'])

        d['stem_model_id'] = torch.tensor(stem_model_id, dtype=torch.long)
        d['stem_spec_id'] = torch.tensor(stem_spec_id, dtype=torch.long)
        d['numerics'] = torch.tensor([cup_outer_value, head_outer_value, head_offset_value, liner_offset_value], dtype=torch.float32)
        d['masks'] = torch.tensor(
            [stem_model_mask, stem_spec_mask, cup_outer_mask, head_outer_mask, head_offset_mask, liner_offset_mask],
            dtype=torch.float32,
        )

        d.pop('context', None)
        return d


def rflow_transforms(image_mean, image_sf, cond_mean, cond_sf):
    return [
        LoadRFlowLatentsd(keys=['image']),
        ScaleLatentd(
            keys=['image', 'condition'],
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        ),
        PrepareProsthesisConditiond(keys=['context']),
    ]


class StructuredProsthesisConditionEncoder(torch.nn.Module):
    """将结构化假体条件编码为 UNet 交叉注意力 token"""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem_model_emb = torch.nn.Embedding(len(FEMORAL_STEM_MODELS), embed_dim)
        self.stem_spec_emb = torch.nn.Embedding(len(FEMORAL_STEM_SPECS), embed_dim)

        self.cup_outer_proj = torch.nn.Linear(1, embed_dim)
        self.head_outer_proj = torch.nn.Linear(1, embed_dim)
        self.head_offset_proj = torch.nn.Linear(1, embed_dim)
        self.liner_offset_proj = torch.nn.Linear(1, embed_dim)
        self.token_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, stem_model_id, stem_spec_id, numerics, masks=None):
        stem_model_embed = self.stem_model_emb(stem_model_id)
        stem_spec_embed = self.stem_spec_emb(stem_spec_id)
        cup_outer_embed = self.cup_outer_proj(numerics[:, 0:1])
        head_outer_embed = self.head_outer_proj(numerics[:, 1:2])
        head_offset_embed = self.head_offset_proj(numerics[:, 2:3])
        liner_offset_embed = self.liner_offset_proj(numerics[:, 3:4])

        out = torch.stack(
            [stem_model_embed, stem_spec_embed, cup_outer_embed, head_outer_embed, head_offset_embed, liner_offset_embed],
            dim=1,
        )
        out = self.token_norm(out)

        if masks is not None:
            masks = enforce_prosthesis_condition_dependencies(masks)
            out = out * masks.unsqueeze(-1)

        return out


def rflow_unet(context_embedding_size=256):
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=8,
        num_res_blocks=(2, 2, 2),
        channels=(96, 192, 384),
        attention_levels=(False, False, True),  # 启用自注意力学习解剖方位关系
        norm_num_groups=32,
        with_conditioning=True,  # 启用交叉注意力注入全局条件
        transformer_num_layers=2,
        cross_attention_dim=context_embedding_size,
        use_flash_attention=True,
    )


def scheduler_rflow():
    return RFlowScheduler(num_train_timesteps=1000)


class EMA:
    """指数移动平均 (Exponential Moving Average) 用于稳定扩散模型的生成质量"""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # 注册模型参数到 shadow 字典
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        """在每个训练 step 后更新 EMA 权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model):
        """暂存当前模型的真实权重 (验证前调用)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()

    @torch.no_grad()
    def copy_to(self, model):
        """将 EMA 权重应用到模型 (验证时调用)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        """恢复模型的真实权重 (验证后调用，继续训练)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data.copy_(self.original[name])
        self.original = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = deepcopy(state_dict)
