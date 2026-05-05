from copy import deepcopy

import numpy as np
import torch
from monai.losses import PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.transforms import (
    CopyItemsd,
    DeleteItemsd,
    DivisiblePadd,
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
    return [
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image'], channel_dim=-1 if channels > 1 else 'no_channel'),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=-1.0),
        DivisiblePadd(keys=['image'], k=16, constant_values=-1.0),
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
    return [
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image'], channel_dim=-1 if channels > 1 else 'no_channel'),
        SpatialPadd(keys=['image'], spatial_size=patch_size, constant_values=-1.0),
        DivisiblePadd(keys=['image'], k=16, constant_values=-1.0),
    ]


class LoadLatentConditiond(MapTransform):
    """读取 .npy 文件 latent 数据 [8, D, H, W] float16"""

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


def ldm_collate_fn(batch):
    """
    Latent 尺寸不一致，ListBatch 策略比 Padding 到固定较大尺寸更好
    list 虽然不是 GPU 并行，但节省了不必要的 Padding 背景计算，同时保留了随机尺寸的泛化能力
    """
    return {
        'image': [item['image'] for item in batch],
        'condition': [item['condition'] for item in batch],
    }


def ldm_transforms(image_mean, image_sf, cond_mean, cond_sf):
    return [
        LoadLatentConditiond(keys=['image']),
        ScaleLatentd(
            keys=['image', 'condition'],
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        ),
    ]


def ldm_unet():
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=8,
        num_res_blocks=(2, 2, 2),
        channels=(64, 128, 256),
        attention_levels=(False, False, True),  # 启用自注意力学习解剖方位关系
        norm_num_groups=32,
        with_conditioning=False,  # TODO 交叉注意力注入全局条件
        use_flash_attention=True,
    )


def scheduler_ddpm():
    return DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        # beta_start=0.00085,  # LDM 标准参数
        # beta_end=0.012,  # LDM 标准参数
    )


def scheduler_ddim():
    return DDIMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        prediction_type='epsilon',
        # beta_start=0.00085,
        # beta_end=0.012,
        clip_sample=False,
    )


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

    def update(self, model):
        """在每个训练 step 后更新 EMA 权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def store(self, model):
        """暂存当前模型的真实权重 (验证前调用)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()

    def copy_to(self, model):
        """将 EMA 权重应用到模型 (验证时调用)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data.copy_(self.shadow[name])

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
