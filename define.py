import json
from copy import deepcopy

import numpy as np
import torch
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


class LoadLatentConditiond(MapTransform):
    """读取 .npy 文件 latent 数据 [12, D, H, W] float16"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        # 加载 npy
        data_npy = np.load(d['image'])

        # 转换为 Tensor
        if isinstance(data_npy, np.ndarray):
            data_tensor = torch.from_numpy(data_npy).float()
        else:
            data_tensor = data_npy.float()

        d['condition'] = data_tensor[0:4]  # 术前
        d['image'] = data_tensor[4:12]  # 假体

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


def generate_text(ctx, level='full'):
    """
    Generate dense K-V string from context.
    Levels:
    - 'full': keep all parameters
    - 'model_size': keep only Femoral Model and Size
    - 'model': keep only Femoral Model
    """
    if not ctx:
        return ''

    def has_value(x):
        return x is not None and x != ''

    parts = []

    femoral_spec = ctx.get('femoral_spec', [])
    femoral_model = femoral_spec[0] if len(femoral_spec) >= 1 and has_value(femoral_spec[0]) else None
    femoral_size = femoral_spec[1] if len(femoral_spec) >= 2 and has_value(femoral_spec[1]) else None
    if femoral_model:
        if level in ['full', 'model_size']:
            if femoral_size:
                parts.append(f'Femoral Model: {femoral_model}, Size: {femoral_size}')
            else:
                parts.append(f'Femoral Model: {femoral_model}')
        elif level == 'model':
            parts.append(f'Femoral Model: {femoral_model}')

    if level == 'full':
        head_outer = ctx.get('head_outer')
        if has_value(head_outer):
            parts.append(f'Head Diameter: {head_outer} mm')

        head_offset = ctx.get('head_offset')
        if has_value(head_offset):
            parts.append(f'Head Offset: {head_offset} mm')

        cup_outer = ctx.get('cup_outer_best')
        if not has_value(cup_outer):
            cup_outer = ctx.get('cup_outer')
        if has_value(cup_outer):
            parts.append(f'Cup Diameter: {cup_outer} mm')

        liner_material = ctx.get('liner_material')
        if has_value(liner_material):
            parts.append(f'Liner Material: {liner_material}')

        liner_offset = ctx.get('liner_offset')
        if has_value(liner_offset):
            parts.append(f'Liner Offset: {liner_offset} mm')

    return ' | '.join(parts)


class PrepareContextd(MapTransform):
    """提取 TOML 中的手术设计参数并保留为原始字典"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        ctx = d.get('context', {})
        d['ctx_raw'] = json.dumps(ctx)

        # 默认这里不再预先生成 c_text，因为训练时会在每个 batch 内动态生成
        d.pop('context', None)

        return d


def rflow_transforms(image_mean, image_sf, cond_mean, cond_sf):
    return [
        LoadLatentConditiond(keys=['image']),
        ScaleLatentd(
            keys=['image', 'condition'],
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        ),
        PrepareContextd(keys=['context']),
    ]


class ContextEmbedder(torch.nn.Module):
    """将带噪参数 c_t 映射为 UNet 交叉注意力 Token 的模态桥梁"""

    def __init__(self, embed_dim=256):
        super().__init__()
        # 初始化一个 3 层 MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 1536))

    def forward(self, c_t):
        # c_t Shape: [B, 768]
        x = self.mlp(c_t)
        # 返回形状为 [B, 6, 256] 的 context_tokens
        out = x.view(x.shape[0], 6, 256)
        return out


class TextEmbeddingNormalizer(torch.nn.Module):
    """将 PubMedBERT 参数向量映射到更适合流匹配的标准化空间"""

    def __init__(self, mean, whitening, coloring):
        super().__init__()
        self.register_buffer('mean', mean.float())
        self.register_buffer('whitening', whitening.float())
        self.register_buffer('coloring', coloring.float())

    @classmethod
    def fit(cls, embeddings, shrinkage=0.05, eps=1e-5):
        x = embeddings.float()
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        cov = centered.T @ centered / max(x.shape[0] - 1, 1)
        avg_var = torch.trace(cov) / cov.shape[0]
        cov = (1.0 - shrinkage) * cov + shrinkage * avg_var * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=eps)
        whitening = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        coloring = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
        return cls(mean.cpu(), whitening.cpu(), coloring.cpu())

    def normalize(self, embeddings):
        return (embeddings.float() - self.mean.to(embeddings.device)) @ self.whitening.to(embeddings.device)

    def denormalize(self, embeddings):
        return embeddings.float() @ self.coloring.to(embeddings.device) + self.mean.to(embeddings.device)

    def forward(self, embeddings):
        return self.normalize(embeddings)


class ParameterVelocityHead(torch.nn.Module):
    """预测参数速度 v_c 的轻量级模块"""

    def __init__(self, in_channels=12, c_dim=768):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(8, 32),
            torch.nn.SiLU(),
            torch.nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(16, 64),
            torch.nn.SiLU(),
            torch.nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(32, 128),
            torch.nn.SiLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.time_proj = torch.nn.Sequential(torch.nn.Linear(1, 128), torch.nn.SiLU())
        self.c_proj = torch.nn.Sequential(torch.nn.Linear(c_dim, 128), torch.nn.SiLU())
        self.fc = torch.nn.Linear(128, c_dim)

    def forward(self, x, timesteps, c_t):
        feat = self.conv(x)
        feat = self.pool(feat).view(x.shape[0], -1)  # [B, 128]
        t_emb = self.time_proj(timesteps.view(-1, 1).float() / 1000.0)
        c_emb = self.c_proj(c_t)
        return self.fc(feat + t_emb + c_emb)


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
