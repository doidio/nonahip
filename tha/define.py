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


class PrepareContextd(MapTransform):
    """提取 TOML 中的手术设计参数并转换为 Tensor"""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.brands = sorted(list(FEMORAL.keys()))
        self.brand_to_id = {b: i for i, b in enumerate(self.brands)}

        all_sizes = set()
        for v in FEMORAL.values():
            all_sizes.update(v)
        self.sizes = sorted(list(all_sizes))
        self.size_to_id = {s: i for i, s in enumerate(self.sizes)}

    def __call__(self, data):
        d = dict(data)
        ctx = d.get('context', {})

        # femoral_spec = ["Brand", "Size"]
        spec = ctx.get('femoral_spec', ['', ''])
        brand = spec[0] if len(spec) > 0 else ''
        size = spec[1] if len(spec) > 1 else ''

        brand_id = self.brand_to_id.get(brand, 0)
        brand_mask = 1.0 if brand in self.brand_to_id else 0.0

        size_id = self.size_to_id.get(size, 0)
        size_mask = 1.0 if size in self.size_to_id else 0.0

        # Numerics: [cup_outer, head_outer, head_offset, liner_offset]
        # 使用 Min-Max 归一化到 [-1, 1] 范围，提高对极端条件的鲁棒性
        def min_max_scale(val, min_val, max_val):
            if val is None or val == '':
                return 0.0, 0.0
            # 线性映射 [min, max] -> [-1, 1]
            return 2.0 * (float(val) - min_val) / (max_val - min_val) - 1.0, 1.0

        cup_outer = ctx['cup_outer_best'] if 'cup_outer_best' in ctx else ctx.get('cup_outer')
        head_outer = ctx.get('head_outer')
        head_offset = ctx.get('head_offset')
        liner_offset = ctx['liner_offset_best'] if 'liner_offset_best' in ctx else ctx.get('liner_offset')

        cup_outer_val, cup_outer_mask = min_max_scale(cup_outer, 38.0, 62.0)
        head_outer_val, head_outer_mask = min_max_scale(head_outer, 22.0, 44.0)
        head_offset_val, head_offset_mask = min_max_scale(head_offset, -5.0, 9.0)
        liner_offset_val, liner_offset_mask = min_max_scale(liner_offset, 0.0, 6.0)

        nums = [cup_outer_val, head_outer_val, head_offset_val, liner_offset_val]
        masks = [brand_mask, size_mask, cup_outer_mask, head_outer_mask, head_offset_mask, liner_offset_mask]

        d['brand_id'] = torch.tensor(brand_id, dtype=torch.long)
        d['size_id'] = torch.tensor(size_id, dtype=torch.long)
        d['numerics'] = torch.tensor(nums, dtype=torch.float32)
        d['masks'] = torch.tensor(masks, dtype=torch.float32)

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
    """将手术设计参数编码为全局条件向量序列 [B, 6, C]"""

    def __init__(self, embed_dim=256):
        super().__init__()
        brands = sorted(list(FEMORAL.keys()))
        all_sizes = set()
        for v in FEMORAL.values():
            all_sizes.update(v)
        sizes = sorted(list(all_sizes))

        self.brand_emb = torch.nn.Embedding(len(brands), embed_dim)
        self.size_emb = torch.nn.Embedding(len(sizes), embed_dim)

        # 为每个数值参数配置独立的投影层
        self.cup_outer_proj = torch.nn.Linear(1, embed_dim)
        self.head_outer_proj = torch.nn.Linear(1, embed_dim)
        self.head_offset_proj = torch.nn.Linear(1, embed_dim)
        self.liner_offset_proj = torch.nn.Linear(1, embed_dim)

    def forward(self, brand_id, size_id, numerics, masks=None):
        brand_embed = self.brand_emb(brand_id)  # [B, C]
        size_embed = self.size_emb(size_id)  # [B, C]

        # 分解并投影数值参数
        cup_outer_embed = self.cup_outer_proj(numerics[:, 0:1])  # [B, C]
        head_outer_embed = self.head_outer_proj(numerics[:, 1:2])  # [B, C]
        head_offset_embed = self.head_offset_proj(numerics[:, 2:3])  # [B, C]
        liner_offset_embed = self.liner_offset_proj(numerics[:, 3:4])  # [B, C]

        # 堆叠成序列，共 6 个 Token [B, 6, C]
        # 注意顺序: brand, size, cup_outer, head_outer, head_offset, liner_offset
        out = torch.stack([brand_embed, size_embed, cup_outer_embed, head_outer_embed, head_offset_embed, liner_offset_embed], dim=1)

        if masks is not None:
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
