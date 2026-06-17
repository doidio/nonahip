import argparse
import json
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SaveImage
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import define
from kernel import fast_drr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # 妥协与性能优化折中说明：
    # 禁用 cudnn.benchmark 是为了避免动态尺寸输入时，cuDNN 频繁检测并重新编译 3D 卷积计算图带来的开销。
    # 因为每个样本的 ROI 尺寸不一致，逐个读取训练时输入尺度在不断变化。
    # 禁用后虽然失去了针对固定尺寸的极致优化，但能彻底根除由于尺寸切换产生的每步卡顿与重编译延迟。
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = tomlkit.loads(config_path.read_text('utf-8')).unwrap()

    train_root = Path(str(cfg['train']['root']))
    dataset_root = Path(cfg['dataset']['root'])
    log_dir = train_root / 'logs'
    ckpt_dir = train_root / 'checkpoints'

    task = 'rflow'
    (
        use_amp,
        num_workers,
        num_epochs,
        val_interval,
        sw_batch_size,
        lr,
        effective_batch_size,
        ema_decay,
    ) = [
        cfg['train'][task][_]
        for _ in (
            'use_amp',
            'num_workers',
            'num_epochs',
            'val_interval',
            'sw_batch_size',
            'lr',
            'effective_batch_size',
            'ema_decay',
        )
    ]

    print('Effective Batch:\t', effective_batch_size)

    patch_size = list(cfg['train']['vae']['patch_size'])

    val_prls, test_prls = set(cfg['val'].keys()), set(cfg['test'].keys())
    train_files, val_files, test_files = [], [], []

    for image_file in (train_root / 'latents').glob('*.npy'):
        prl = '_'.join(image_file.name.removesuffix('.npy').split('_')[:2])
        if prl in cfg['pairs']['excluded']:
            continue

        pid, rl = prl.split('_')
        f = dataset_root / 'pair' / pid / rl / 'context.toml'
        if f.exists():
            it = {'image': image_file.as_posix(), 'prl': prl, 'context': tomlkit.loads(f.read_text('utf-8')).unwrap()}
        else:
            raise RuntimeError(f'Non-exist {f.as_posix()}')

        if prl in test_prls:
            test_files.append(it)
        elif prl in val_prls:
            val_files.append(it)
        else:
            train_files.append(it)

    train_files.sort(key=lambda x: x['prl'])
    val_files.sort(key=lambda x: x['prl'])
    test_files.sort(key=lambda x: x['prl'])

    val_prl = val_files[0]['prl'] if len(val_files) else None

    print('Train:\t', len(train_files))
    print('Val:\t', len(val_files))

    def load_vae(subtask):
        ckpt_path = (ckpt_dir / f'vae_{subtask}_best.pt').resolve()

        print(f'[{subtask}]\t', f'Loading {ckpt_path}')

        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)

        print('Epoch:\t', loaded['epoch'])
        print('Channels:\t', channels := loaded['channels'])
        print('L1:   \t', loaded['val_l1'], 'best', loaded['best_val_l1'])
        print('PSNR:\t', loaded['val_psnr'])
        print('SSIM:\t', loaded['val_ssim'])
        print('Scale Factor:\t', sf := loaded['scale_factor'])
        print('Global Mean:\t', mean := loaded['global_mean'])

        vae = define.vae_kl(channels).to(device)
        vae.load_state_dict(loaded['state_dict'])
        vae.eval().float()
        print('Param:\t {0:.2f} B'.format(sum(p.numel() for p in vae.parameters()) / 1e9))

        i_val, r_val = 0.0, 0.0
        for metric in ('FID', 'Eikonal'):
            kw = f'i{metric.lower()}'
            if kw in loaded:
                print(f'i{metric}:\t', i_val := loaded[kw])
            kw = f'r{metric.lower()}'
            if kw in loaded:
                print(f'r{metric}:\t', r_val := loaded[kw])
        print('Interp/Recon:\t', i_val / (r_val + 1e-12))

        return vae, sf, mean

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

    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)

    # 妥协与机制调整说明：
    # 为了避免批次间的动态 Padding (导致背景噪声区占比较大以及引入人为边界伪影)，
    # 我们将 batch_size 设为 1，逐个加载单样本进行训练。
    # 由于每个样本单独加载，不再存在多样本拼 Batch 时的尺寸对齐需求，故完全取消了动态 Padding 整理函数 (collate_fn)
    # 和动态体积采样器 (batch_sampler)。
    # 显存及优化稳定性方面，通过梯度累加 (每 effective_batch_size=12 次反向传播后执行一次优化器更新)
    # 依然能够实现宏观上的大 Batch 均值效应，确保训练的稳定性与收敛效果。
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    # 验证 Loader 保持 BS=1 即可
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    embed_dim = 256
    rflow = define.rflow_unet(context_embedding_size=embed_dim).to(device)
    context_embedder = define.ContextEmbedder(embed_dim=embed_dim).to(device)
    param_head = define.ParameterVelocityHead().to(device)
    rflow_ema = define.EMA(rflow, decay=ema_decay)
    context_ema = define.EMA(context_embedder, decay=ema_decay)
    param_ema = define.EMA(param_head, decay=ema_decay)

    print('Loading Sentence-PubMedBERT...')
    model_dir = cfg['pretrained']['text_encoder']
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text_encoder = AutoModel.from_pretrained(model_dir).to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    def encode_texts(texts):
        tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = text_encoder(**tokens)
            attn_mask = tokens['attention_mask']
            token_embs = outputs.last_hidden_state
            input_mask = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
            return torch.sum(token_embs * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

    print('Fitting PubMedBERT embedding normalizer...')
    train_texts = [define.generate_text(it['context'], level='full') for it in train_files]
    normalizer_texts = []
    for it in train_files:
        ctx = it['context']
        full_text = define.generate_text(ctx, level='full')
        normalizer_texts.extend(['', '', full_text, full_text, define.generate_text(ctx, level='model'), define.generate_text(ctx, level='model_size')])
    normalizer_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(normalizer_texts), 64), desc='TextNorm'):
            normalizer_embeddings.append(encode_texts(normalizer_texts[i : i + 64]).cpu())
        text_normalizer = define.TextEmbeddingNormalizer.fit(torch.cat(normalizer_embeddings, dim=0), shrinkage=0.005, eps=1e-4).to(device)
        train_full_embeddings = []
        for i in range(0, len(train_texts), 64):
            train_full_embeddings.append(encode_texts(train_texts[i : i + 64]).cpu())
        train_full_embeddings = torch.cat(train_full_embeddings, dim=0).to(device)
        c_noise_scale = torch.linalg.vector_norm(text_normalizer(train_full_embeddings), dim=1).mean() / (train_full_embeddings.shape[1] ** 0.5)
    print('Parameter noise scale:\t', float(c_noise_scale))

    scheduler = define.scheduler_rflow()

    optimizer = torch.optim.AdamW(
        list(rflow.parameters()) + list(context_embedder.parameters()) + list(param_head.parameters()), lr=lr, weight_decay=1e-5
    )

    scaler = GradScaler() if use_amp else None

    start_epoch = 0

    # 继续训练
    if args.resume:
        load_pt = (ckpt_dir / f'{task}_last.pt').resolve()
    else:
        load_pt = None

    if load_pt and load_pt.exists():
        try:
            print('Resuming:\t', load_pt)
            ckpt = torch.load(load_pt, map_location=device)
            rflow.load_state_dict(ckpt['rflow_state'])

            if 'context_state' in ckpt:
                print('Loading ContextEmbedder...')
                context_embedder.load_state_dict(ckpt['context_state'])

            if 'param_state' in ckpt:
                print('Loading ParameterVelocityHead...')
                param_head.load_state_dict(ckpt['param_state'])

            if 'text_normalizer' in ckpt:
                print('Loading TextEmbeddingNormalizer...')
                text_normalizer.load_state_dict(ckpt['text_normalizer'])
            if 'c_noise_scale' in ckpt:
                c_noise_scale = torch.as_tensor(ckpt['c_noise_scale'], device=device)
                print('Parameter noise scale:\t', float(c_noise_scale))

            optimizer.load_state_dict(ckpt['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = lr

            if 'rflow_state_ema' in ckpt:
                rflow_ema.load_state_dict(ckpt['rflow_state_ema'])

            if 'context_state_ema' in ckpt:
                context_ema.load_state_dict(ckpt['context_state_ema'])

            if 'param_state_ema' in ckpt:
                param_ema.load_state_dict(ckpt['param_state_ema'])

            if use_amp and 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])

            start_epoch = ckpt['epoch']
            val_loss = ckpt.get('val_loss', float('inf'))

            # Explicitly delete the loaded checkpoint to free up system/GPU memory
            del ckpt
            torch.cuda.empty_cache()

            print('Epoch:\t', start_epoch)
            print('MSE:\t', val_loss)
            start_epoch += 1
        except Exception as e:
            raise SystemError(f'Load failed: {e}')

    # 日志
    if args.resume:
        candidates = []
        if log_dir.exists():
            for p in log_dir.iterdir():
                if p.is_dir() and p.name.startswith(f'{task}_'):
                    candidates.append(p)
        if candidates:
            # 根据目录名中的时间戳排序 (如 rflow_20260609_160842)
            prefix = f'{task}_'

            def get_sort_key(p):
                name = p.name
                if name.startswith(prefix):
                    ts = name[len(prefix) : len(prefix) + 15]
                    parts = ts.split('_')
                    if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit() and len(parts[1]) == 6:
                        return (ts, name)
                return ('', p.name)

            candidates.sort(key=get_sort_key)
            log_dir = candidates[-1]
            print('Resuming logs in:\t', log_dir)
        else:
            suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S_resume')
            log_dir = log_dir / suffix
            print('No existing log directory found for resume. Creating:\t', log_dir)
    else:
        suffix = datetime.now().strftime(f'{task}_%Y%m%d_%H%M%S')
        log_dir = log_dir / suffix

    writer = SummaryWriter(log_dir=log_dir.as_posix())

    saver = SaveImage(
        output_dir=log_dir,
        output_postfix='',
        output_ext='.nii.gz',
        separate_folder=False,
        print_log=False,
        resample=False,
    )

    def decode(z, name, vae_model, sf, mean, ep):
        z = (z / sf + mean).detach().to(device).float()

        def decode_predictor(inputs: torch.Tensor) -> torch.Tensor:
            vae_latent_ch = vae_model.latent_channels
            if inputs.shape[1] > vae_latent_ch:
                recons = []
                for i in range(0, inputs.shape[1], vae_latent_ch):
                    recons.append(vae_model.decode(inputs[:, i : i + vae_latent_ch]))
                return torch.cat(recons, dim=1)
            return vae_model.decode(inputs)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=False):
                recon = sliding_window_inference(
                    inputs=z,
                    roi_size=[p // define.vae_downsample for p in patch_size],
                    sw_batch_size=sw_batch_size,
                    predictor=decode_predictor,
                    overlap=0.25,
                    mode='gaussian',
                    device=device,
                    sw_device=device,
                    progress=False,
                )

        saver(recon[0].cpu(), meta_data={'filename_or_obj': f'{name}.nii.gz'})
        return recon.cpu()

    amp_ctx = autocast(device.type) if use_amp else nullcontext()

    accumulated_samples = 0
    optimizer.zero_grad(set_to_none=True)

    def optimizer_step(accumulated_count):
        if accumulated_count <= 0:
            return

        if use_amp:
            scaler.unscale_(optimizer)

        # Losses are scaled by effective_batch_size during accumulation; rescale
        # the tail step so a partial final micro-batch still becomes a true mean.
        tail_scale = effective_batch_size / accumulated_count
        for param in list(rflow.parameters()) + list(context_embedder.parameters()) + list(param_head.parameters()):
            if param.grad is not None:
                param.grad *= tail_scale

        torch.nn.utils.clip_grad_norm_(list(rflow.parameters()) + list(context_embedder.parameters()) + list(param_head.parameters()), 1.0)

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            step_skipped = scale_before > scale_after
        else:
            optimizer.step()
            step_skipped = False

        if not step_skipped:
            rflow_ema.update(rflow)
            context_ema.update(context_embedder)
            param_ema.update(param_head)

        optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, num_epochs):
        rflow.train()
        context_embedder.train()
        param_head.train()
        epoch_loss_y = 0
        epoch_loss_c = 0
        step = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs - 1}')

        for batch in pbar:
            step += 1

            image = batch['image'].to(device, non_blocking=True)
            cond = batch['condition'].to(device, non_blocking=True)
            current_bs = image.shape[0]

            c_text_strs_true = []
            c_text_strs_cond = []

            for i in range(current_bs):
                prob = random.random()

                ctx_str = batch['ctx_raw'][i] if 'ctx_raw' in batch else '{}'
                ctx = json.loads(ctx_str)
                true_text = define.generate_text(ctx, level='full')
                c_text_strs_true.append(true_text)

                if prob < 1 / 6:
                    # 1/6: Drop Both (p(y,c))
                    cond[i] = 0.0
                    c_text_strs_cond.append('')
                elif prob < 2 / 6:
                    # 1/6: Drop c only (p(y,c|x))
                    c_text_strs_cond.append('')
                elif prob < 3 / 6:
                    # 1/6: Drop x only (p(y,c|c_full))
                    cond[i] = 0.0
                    c_text_strs_cond.append(true_text)
                elif prob < 4 / 6:
                    # 1/6: Partial c - Model only (p(y,c|x, c_model))
                    c_text_strs_cond.append(define.generate_text(ctx, level='model'))
                elif prob < 5 / 6:
                    # 1/6: Partial c - Model & Size (p(y,c|x, c_model_size))
                    c_text_strs_cond.append(define.generate_text(ctx, level='model_size'))
                else:
                    # 1/6: Keep Both (p(y,c|x, c_full))
                    c_text_strs_cond.append(true_text)

            with torch.no_grad():
                c_text_true = text_normalizer(encode_texts(c_text_strs_true))
                c_text_cond = text_normalizer(encode_texts(c_text_strs_cond))

            with amp_ctx:
                # 采样时间步
                timesteps = scheduler.sample_timesteps(image)

                # 获取加噪比例
                t_val = (timesteps.float() / scheduler.num_train_timesteps).view(-1, 1).to(device)

                # 设计妥协说明：移除了 valid_mask。因为训练数据已在离线对齐 32 整数倍，且 batch_size = 1，无批次内 padding。
                noise_y = torch.randn_like(image)

                # RFM 加噪过程
                noisy_image = scheduler.add_noise(original_samples=image, noise=noise_y, timesteps=timesteps)

                # 参数流状态始终由完整真实参数加噪得到；随机丢弃后的参数仅作为条件注入 UNet。
                noise_c = torch.randn_like(c_text_true) * c_noise_scale
                c_t = t_val * c_text_true + (1.0 - t_val) * noise_c

                # 生成全局条件 Embeddings [B, 6, C]
                context = context_embedder(c_text_cond)

                # 拼接输入 (Image + Pre-op Condition)
                input_tensor = torch.cat([noisy_image, cond], dim=1)

                # 预测速度 (Velocity), 注入 Context
                velocity_y_pred = rflow(x=input_tensor, timesteps=timesteps, context=context)
                velocity_c_pred = param_head(x=input_tensor, timesteps=timesteps, c_t=c_t)

                # 计算目标速度 (预测真实的 c_text_true)
                target_velocity_y = image - noise_y
                target_velocity_c = c_text_true - noise_c

                # 联合损失函数计算
                loss_y = torch.nn.functional.mse_loss(velocity_y_pred.float(), target_velocity_y.float(), reduction='mean')
                loss_c = torch.nn.functional.mse_loss(velocity_c_pred.float(), target_velocity_c.float(), reduction='mean')

                # 将两者简单相加，纯粹是为了让 PyTorch 能在一次 backward() 中同时向两个完全独立的网络派发梯度，提升计算效率，数值本身无物理意义
                loss = loss_y + loss_c
                disp_loss_y = loss_y.item()
                disp_loss_c = loss_c.item()

                # 动态梯度累积缩放 (根据当前真实 bs 与期望有效 bs 的比例缩放 loss)
                micro_loss = loss * (current_bs / effective_batch_size)

                if use_amp:
                    scaler.scale(micro_loss).backward()
                else:
                    micro_loss.backward()

            accumulated_samples += current_bs

            if accumulated_samples >= effective_batch_size:
                optimizer_step(accumulated_samples)
                accumulated_samples = 0

            epoch_loss_y += disp_loss_y
            epoch_loss_c += disp_loss_c

            if step % 1 == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar('train/loss_y', disp_loss_y, global_step)
                writer.add_scalar('train/loss_c', disp_loss_c, global_step)

            pbar.set_postfix({'loss_y': f'{disp_loss_y:.4f}', 'loss_c': f'{disp_loss_c:.4f}'})

        # Flush the final partial accumulation before validation/checkpointing.
        if accumulated_samples > 0:
            optimizer_step(accumulated_samples)
            accumulated_samples = 0

        writer.add_scalar('train/epoch_loss_y', epoch_loss_y / step, epoch)
        writer.add_scalar('train/epoch_loss_c', epoch_loss_c / step, epoch)

        # 验证与采样 (保持 BS=1，不需要改 collate_fn)
        if epoch % val_interval == 0:
            rflow.eval()
            context_embedder.eval()
            param_head.eval()
            rflow_ema.store(rflow)
            rflow_ema.copy_to(rflow)
            context_ema.store(context_embedder)
            context_ema.copy_to(context_embedder)
            param_ema.store(param_head)
            param_ema.copy_to(param_head)

            val_loss_y_sum = 0
            val_loss_c_sum = 0
            val_cos_sim_v_sum = 0
            val_cos_sim_vt_sum = 0
            val_cos_sim_v_raw_sum = 0
            val_cos_sim_vt_raw_sum = 0
            val_steps = 0

            with torch.no_grad():
                for val_idx, batch in enumerate(val_bar := tqdm(val_loader, desc='Val')):
                    image = batch['image'].to(device)
                    cond = batch['condition'].to(device)

                    c_text_strs = []
                    for b in range(image.shape[0]):
                        ctx_str = batch['ctx_raw'][b] if 'ctx_raw' in batch else '{}'
                        ctx = json.loads(ctx_str)
                        c_text_strs.append(define.generate_text(ctx, level='full'))

                    with torch.no_grad():
                        c_text_raw = encode_texts(c_text_strs)
                        c_text = text_normalizer(c_text_raw)

                    timesteps = scheduler.sample_timesteps(image)
                    t_val = (timesteps.float() / scheduler.num_train_timesteps).view(-1, 1).to(device)

                    # 设计妥协说明：验证亦移除 valid_mask
                    noise_y = torch.randn_like(image)
                    noisy_image = scheduler.add_noise(original_samples=image, noise=noise_y, timesteps=timesteps)

                    noise_c = torch.randn_like(c_text) * c_noise_scale
                    c_t = t_val * c_text + (1.0 - t_val) * noise_c

                    context = context_embedder(c_text)
                    input_tensor = torch.cat([noisy_image, cond], dim=1)

                    with amp_ctx:
                        velocity_y_pred = rflow(input_tensor, timesteps, context=context)
                        velocity_c_pred = param_head(input_tensor, timesteps, c_t)

                        target_velocity_y = image - noise_y
                        target_velocity_c = c_text - noise_c

                        loss_y = torch.nn.functional.mse_loss(velocity_y_pred.float(), target_velocity_y.float(), reduction='mean')
                        loss_c = torch.nn.functional.mse_loss(velocity_c_pred.float(), target_velocity_c.float(), reduction='mean')

                    val_loss_y_sum += loss_y.item()
                    val_loss_c_sum += loss_c.item()
                    val_steps += 1

                    # 瞬时预测 c_data：基于 Flow Matching 的常微分方程式闭式解 (无需 50 步积分)
                    c_pred = c_t + (1.0 - t_val) * velocity_c_pred
                    batch_cos_sim_v = torch.nn.functional.cosine_similarity(c_pred, c_text, dim=1).mean().item()
                    val_cos_sim_v_sum += batch_cos_sim_v
                    batch_cos_sim_v_raw = torch.nn.functional.cosine_similarity(text_normalizer.denormalize(c_pred), c_text_raw, dim=1).mean().item()
                    val_cos_sim_v_raw_sum += batch_cos_sim_v_raw

                    # 50步精确评估 param_head：从随机 c_t 出发，沿真实 y_t 轨迹积分到 c_data。
                    with torch.no_grad():
                        scheduler.set_timesteps(num_inference_steps=50)
                        temp_timesteps = scheduler.timesteps.to(device)
                        temp_next_timesteps = torch.cat([temp_timesteps[1:], torch.zeros(1, dtype=temp_timesteps.dtype, device=device)])

                        c_t_pred = torch.randn_like(c_text, generator=torch.Generator(device=device).manual_seed(42 + val_idx)) * c_noise_scale

                        for t_s, next_t_s in zip(temp_timesteps, temp_next_timesteps):
                            t_val_s = (t_s.float() / scheduler.num_train_timesteps).view(-1, 1).to(device)
                            y_t_s = t_val_s * image + (1.0 - t_val_s) * noise_y
                            with amp_ctx:
                                v_c_pred_s = param_head(torch.cat([y_t_s, cond], dim=1), t_s[None].to(device).repeat(image.shape[0]), c_t_pred)
                            c_t_pred, _ = scheduler.step(v_c_pred_s, t_s, c_t_pred, next_t_s)

                    batch_cos_sim_vt = torch.nn.functional.cosine_similarity(c_t_pred, c_text, dim=1).mean().item()
                    val_cos_sim_vt_sum += batch_cos_sim_vt
                    batch_cos_sim_vt_raw = torch.nn.functional.cosine_similarity(text_normalizer.denormalize(c_t_pred), c_text_raw, dim=1).mean().item()
                    val_cos_sim_vt_raw_sum += batch_cos_sim_vt_raw

                    prl = batch['prl'][0]
                    if prl == val_prl:
                        name = f'{prl}_{val_idx}'

                        scheduler.set_timesteps(num_inference_steps=50)
                        all_timesteps = scheduler.timesteps
                        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

                        generator = torch.Generator(device=device).manual_seed(42)
                        # 设计妥协说明：移除 valid_mask 遮罩
                        generated = torch.randn(image.shape, device=device, generator=generator)
                        generated_c = torch.randn(c_text.shape, device=device, generator=generator) * c_noise_scale

                        for t, next_t in zip(all_timesteps, all_next_timesteps):
                            val_bar.set_postfix({'RFlow': t.item()})

                            with torch.no_grad(), amp_ctx:  # 使用 AMP 保护以减少显存占用并加速推理
                                t_input = t[None].to(device)

                                current_context = context_embedder(c_text)
                                model_input = torch.cat([generated, cond], dim=1)

                                velocity_y_pred = rflow(model_input, t_input, context=current_context)
                                velocity_c_pred = param_head(model_input, t_input, generated_c)

                            with torch.no_grad():
                                generated, _ = scheduler.step(velocity_y_pred, t, generated, next_t)
                                generated_c, _ = scheduler.step(velocity_c_pred, t, generated_c, next_t)

                        val_bar.set_postfix({})

                        with amp_ctx:
                            vis_generated = decode(generated, f'{name}_val_epoch_{epoch:03d}_Gen', vae_image, image_sf, image_mean, epoch)
                            vis_gt = decode(image, f'{name}_val_epoch_{epoch:03d}_GT', vae_image, image_sf, image_mean, epoch)
                            vis_cond = decode(cond, f'{name}_val_epoch_{epoch:03d}_Cond', vae_cond, cond_sf, cond_mean, epoch)

                        # DRR Visualization (Refer to VAE style)
                        axis = 1
                        val_vis_dir = log_dir / 'val'
                        val_vis_dir.mkdir(parents=True, exist_ok=True)

                        def get_drr_hstack(vis_tensor):
                            drrs = []
                            for c in range(vis_tensor.shape[1]):
                                img = vis_tensor[0, c].numpy()
                                drr = fast_drr(img + 1.0, axis, th=(0.1, 2.0), mode='mean')
                                drrs.append(np.flipud(drr.transpose(1, 0, 2)))
                            return np.hstack(drrs)

                        drr_gen = get_drr_hstack(vis_generated)
                        drr_gt = get_drr_hstack(vis_gt)
                        drr_cond = get_drr_hstack(vis_cond)

                        writer.add_image(f'val/{name}_Gen', drr_gen, epoch, dataformats='HWC')
                        writer.add_image(f'val/{name}_GT', drr_gt, epoch, dataformats='HWC')
                        writer.add_image(f'val/{name}_Cond', drr_cond, epoch, dataformats='HWC')

                        Image.fromarray(drr_gen).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_Gen.png')
                        Image.fromarray(drr_gt).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_GT.png')
                        Image.fromarray(drr_cond).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_Cond.png')

                        # Diff DRR (hstack)
                        diff_drrs = []
                        for c in range(vis_generated.shape[1]):
                            diff = np.abs(vis_generated[0, c].numpy() - vis_gt[0, c].numpy())
                            drr_diff = fast_drr(diff + 1.0, axis, th=(0.1, 2.0), mode='mean')
                            diff_drrs.append(np.flipud(drr_diff.transpose(1, 0, 2)))

                        drr_diff_hstack = np.hstack(diff_drrs)
                        writer.add_image(f'val/Diff_{val_idx}', drr_diff_hstack, epoch, dataformats='HWC')
                        Image.fromarray(drr_diff_hstack).save(val_vis_dir / f'{name}_val_epoch_{epoch:03d}_Diff.png')

            rflow_ema.restore(rflow)
            context_ema.restore(context_embedder)
            param_ema.restore(param_head)
            val_loss_y_avg = val_loss_y_sum / val_steps
            val_loss_c_avg = val_loss_c_sum / val_steps
            val_cos_sim_v_avg = val_cos_sim_v_sum / val_steps
            val_cos_sim_vt_avg = val_cos_sim_vt_sum / val_steps
            val_cos_sim_v_raw_avg = val_cos_sim_v_raw_sum / val_steps
            val_cos_sim_vt_raw_avg = val_cos_sim_vt_raw_sum / val_steps
            writer.add_scalar('val/loss_y', val_loss_y_avg, epoch)
            writer.add_scalar('val/loss_c', val_loss_c_avg, epoch)
            writer.add_scalar('val/c_cossim_v', val_cos_sim_v_avg, epoch)
            writer.add_scalar('val/c_cossim_vt', val_cos_sim_vt_avg, epoch)
            writer.add_scalar('val/c_cossim_v_raw', val_cos_sim_v_raw_avg, epoch)
            writer.add_scalar('val/c_cossim_vt_raw', val_cos_sim_vt_raw_avg, epoch)

            print(
                f'Val Loss Y: {val_loss_y_avg:11.4f} | Val Loss C: {val_loss_c_avg:11.4f} | '
                f'Val CosSim(V): {val_cos_sim_v_avg:11.4f} | Val CosSim(Vt): {val_cos_sim_vt_avg:11.4f} | '
                f'Raw CosSim(V): {val_cos_sim_v_raw_avg:11.4f} | Raw CosSim(Vt): {val_cos_sim_vt_raw_avg:11.4f}'
            )

            ckpt = {
                'epoch': epoch,
                'rflow_state': rflow.state_dict(),
                'rflow_state_ema': rflow_ema.state_dict(),
                'context_state': context_embedder.state_dict(),
                'context_state_ema': context_ema.state_dict(),
                'param_state': param_head.state_dict(),
                'param_state_ema': param_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'text_normalizer': text_normalizer.state_dict(),
                'c_noise_scale': float(c_noise_scale.detach().cpu()),
                'val_loss_y': val_loss_y_avg,
                'val_loss_c': val_loss_c_avg,
                'val_c_cossim_v': val_cos_sim_v_avg,
                'val_c_cossim_vt': val_cos_sim_vt_avg,
                'val_c_cossim_v_raw': val_cos_sim_v_raw_avg,
                'val_c_cossim_vt_raw': val_cos_sim_vt_raw_avg,
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(ckpt, ckpt_dir / f'{task}_last.pt')

        torch.cuda.empty_cache()

    writer.close()
    print('Training Completed.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupted terminating...')
