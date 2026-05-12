import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np


def _printf(*args):
    print(*args)


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
        channels=(64, 128, 256),
        attention_levels=(False, False, True),
        norm_num_groups=32,
        with_conditioning=False,
        use_flash_attention=not cpu,
    ).to(device)

    loaded = torch.load(rflow_path, map_location=device)
    printf('Epoch:\t {0}'.format(loaded['epoch']))
    printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in rflow_model.parameters()) / 1e9))

    if 'ema_state' in loaded:
        rflow_model.load_state_dict(loaded['ema_state'])
        printf('Loaded EMA weights.')
    else:
        rflow_model.load_state_dict(loaded['state_dict'])

    rflow_model.eval().float()

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
                    velocity_pred = rflow_model(model_input, t[None].to(device))
            else:
                # 1-RF Mode or Multi-step: CFG logic
                t_input = t[None].to(device).repeat(2)

                if cfg_val > 1.0:
                    latent_input = torch.cat([generated] * 2, dim=0)
                    cond_input = torch.cat([cond_encoded, torch.zeros_like(cond_encoded)], dim=0)
                    model_input = torch.cat([latent_input, cond_input], dim=1)

                    with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                        velocity_pred_batch = rflow_model(model_input, t_input)

                    velocity_cond, velocity_uncond = velocity_pred_batch.chunk(2)
                    velocity_pred = velocity_uncond + cfg_val * (velocity_cond - velocity_uncond)
                elif cfg_val == 1.0:
                    model_input = torch.cat([generated, cond_encoded], dim=1)
                    with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                        velocity_pred = rflow_model(model_input, t[None].to(device))
                else:
                    model_input = torch.cat([generated, torch.zeros_like(cond_encoded)], dim=1)
                    with autocast(device.type) if amp and device.type != 'cpu' else nullcontext():
                        velocity_pred = rflow_model(model_input, t[None].to(device))

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
