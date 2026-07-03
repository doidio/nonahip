"""
Train a metal geometry cls model in real, generated, or extended domain.

Real-domain cls:
    real metal latent -> prosthesis labels, validated on real data and
    optionally on RFlow-generated p(y | c) samples.

Generated-domain cls:
    RFlow-generated p(y | c) metal latent -> prosthesis labels, validated on both
    generated and real data. The gap between the two domains is intended for
    downstream safety analysis of free p(y | x) samples.

Extended-domain cls:
    RFlow-generated metal latent from dynamically sampled prosthesis parameters.
    This branch is intended for parameter-space extension beyond the observed
    clinical frequency distribution.
"""

import argparse
import itertools
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, MapTransform
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import define

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_NAMES = ('stem_model', 'stem_spec', 'cup_outer', 'head_outer', 'head_offset', 'liner_offset')
METRIC_NAMES = LABEL_NAMES + ('stem_spec_global', 'all_labeled_exact')
NUMERIC_LABEL_NAMES = ('cup_outer', 'head_outer', 'head_offset', 'liner_offset')
NUMERIC_CLASS_STEPS = {
    'cup_outer': 2.0,
    'head_outer': 2.0,
    'head_offset': 0.5,
}


def _valid_value(value):
    return define._has_context_value(value)


def _context_number(ctx, key):
    value = ctx.get('liner_offset') if key == 'liner_offset' else define._get_context_value(ctx, key)
    if not _valid_value(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def build_numeric_bins():
    bins = {}
    for name in NUMERIC_LABEL_NAMES:
        if name == 'liner_offset':
            bins[name] = (0.0, 4.0)
            continue
        min_value, max_value = define.PROSTHESIS_NUMERIC_RANGES[name]
        step = NUMERIC_CLASS_STEPS[name]
        bins[name] = tuple(float(x) for x in np.arange(min_value, max_value + 1e-6, step))
    return bins


def nearest_numeric_id(value, values):
    values_np = np.asarray(values, dtype=np.float32)
    return int(np.abs(values_np - float(value)).argmin())


def build_stem_spec_model_mask(device):
    mask = torch.zeros((len(define.FEMORAL_STEM_MODELS), len(define.FEMORAL_STEM_SPECS)), dtype=torch.bool, device=device)
    model_to_id = {model: i for i, model in enumerate(define.FEMORAL_STEM_MODELS)}
    for spec_id, (model, size) in enumerate(define.FEMORAL_STEM_SPECS):
        if define._has_context_value(model) and define._has_context_value(size):
            mask[model_to_id[model], spec_id] = True
    return mask


def extract_label_ids(ctx, numeric_class_values, condition_transform):
    encoded = condition_transform({'context': ctx})
    numeric_ids = []
    numeric_masks = []
    for name in NUMERIC_LABEL_NAMES:
        value = _context_number(ctx, name)
        if value is None:
            numeric_ids.append(0)
            numeric_masks.append(0.0)
        else:
            numeric_ids.append(nearest_numeric_id(value, numeric_class_values[name]))
            numeric_masks.append(1.0)
    encoded['numeric_class_ids'] = torch.tensor(numeric_ids, dtype=torch.long)
    encoded['numeric_class_masks'] = torch.tensor(numeric_masks, dtype=torch.float32)
    return encoded


def build_split_files(cfg, include_test=False):
    train_root = Path(str(cfg['train']['root']))
    dataset_root = Path(cfg['dataset']['root'])
    excluded = set(cfg['pairs']['excluded'])
    val_prls = set(cfg['val'].keys())
    test_prls = set(cfg['test'].keys())

    train_files, val_files, test_files, all_files = [], [], [], []
    for image_file in sorted((train_root / 'latents').glob('*.npy')):
        prl = '_'.join(image_file.name.removesuffix('.npy').split('_')[:2])
        if prl in excluded:
            continue
        if prl in test_prls and not include_test:
            split = 'test'
        elif prl in val_prls:
            split = 'val'
        elif prl in test_prls:
            split = 'test'
        else:
            split = 'train'

        pid, rl = prl.split('_')
        context_path = dataset_root / 'pair' / pid / rl / 'context.toml'
        if not context_path.exists():
            raise RuntimeError(f'Non-exist {context_path.as_posix()}')
        item = {'image': image_file.as_posix(), 'prl': prl, 'context': tomlkit.loads(context_path.read_text('utf-8')).unwrap()}
        all_files.append(item)
        if split == 'train':
            train_files.append(item)
        elif split == 'val':
            val_files.append(item)
        elif include_test:
            test_files.append(item)

    return train_files, val_files, test_files, all_files


def latent_voxel_count(item):
    shape = np.load(item['image'], mmap_mode='r').shape
    if len(shape) < 4:
        raise RuntimeError(f'Unexpected latent shape {shape}: {item["image"]}')
    return int(np.prod(shape[1:]))


def select_largest_latent_files(files, count):
    if count <= 0:
        return []
    return sorted(files, key=latent_voxel_count, reverse=True)[: min(count, len(files))]


def load_vae_stats(ckpt_dir, subtask):
    loaded = torch.load((ckpt_dir / f'vae_{subtask}_best.pt').resolve(), map_location='cpu', weights_only=False)
    return loaded['scale_factor'], loaded['global_mean']


class PrepareProsthesisClsLabeld(MapTransform):
    def __init__(self, keys, numeric_class_values, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.condition_transform = define.PrepareProsthesisConditiond(keys=keys)
        self.numeric_class_values = numeric_class_values

    def __call__(self, data):
        d = dict(data)
        ctx = d.get('context', {})
        d.update(extract_label_ids(ctx, self.numeric_class_values, self.condition_transform))
        return d


def cls_transforms(image_mean, image_sf, cond_mean, cond_sf, numeric_class_values):
    return [
        define.LoadRFlowLatentsd(keys=['image']),
        define.ScaleLatentd(
            keys=['image', 'condition'],
            image_mean=image_mean,
            image_sf=image_sf,
            cond_mean=cond_mean,
            cond_sf=cond_sf,
        ),
        PrepareProsthesisClsLabeld(keys=['context'], numeric_class_values=numeric_class_values),
    ]


def _normalized_positive(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values > 0
    if positive.any():
        values[positive] = values[positive] / values[positive].mean()
    return values


def build_balanced_train_sampler(files, numeric_class_values):
    condition_transform = define.PrepareProsthesisConditiond(keys=['context'])
    labels = [extract_label_ids(item['context'], numeric_class_values, condition_transform) for item in files]

    def inverse_frequency(ids, masks):
        counts = {}
        for item_id, item_mask in zip(ids, masks):
            if item_mask > 0:
                counts[item_id] = counts.get(item_id, 0) + 1
        return [1.0 / counts[item_id] if item_mask > 0 else 0.0 for item_id, item_mask in zip(ids, masks)]

    model_ids = [int(label['stem_model_id']) for label in labels]
    model_masks = [float(label['masks'][0]) for label in labels]
    model_weights = _normalized_positive(inverse_frequency(model_ids, model_masks))

    spec_ids = [int(label['stem_spec_id']) for label in labels]
    spec_masks = [float(label['masks'][1]) for label in labels]
    spec_weights = _normalized_positive(inverse_frequency(spec_ids, spec_masks))

    numeric_weights_by_name = []
    for idx, _ in enumerate(NUMERIC_LABEL_NAMES):
        ids = [int(label['numeric_class_ids'][idx]) for label in labels]
        masks = [float(label['numeric_class_masks'][idx]) for label in labels]
        numeric_weights_by_name.append(np.asarray(inverse_frequency(ids, masks), dtype=np.float64))
    numeric_weights = np.stack(numeric_weights_by_name, axis=1)
    numeric_valid = numeric_weights > 0
    numeric_group = np.divide(
        numeric_weights.sum(axis=1),
        np.maximum(numeric_valid.sum(axis=1), 1),
        out=np.zeros(len(files), dtype=np.float64),
        where=numeric_valid.sum(axis=1) > 0,
    )
    numeric_group = _normalized_positive(numeric_group)

    group_weights = np.stack([model_weights, spec_weights, numeric_group], axis=1)
    group_valid = group_weights > 0
    sample_weights = np.divide(
        group_weights.sum(axis=1),
        np.maximum(group_valid.sum(axis=1), 1),
        out=np.ones(len(files), dtype=np.float64),
        where=group_valid.sum(axis=1) > 0,
    )
    sample_weights = np.clip(sample_weights, 1e-6, None)
    return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(files), replacement=True)


class ExtendedLabelSampler:
    def __init__(self, numeric_class_values):
        self.stem_specs = tuple(
            spec for spec in define.FEMORAL_STEM_SPECS if define._has_context_value(spec[0]) and define._has_context_value(spec[1])
        )
        if not self.stem_specs:
            raise RuntimeError('Empty femoral stem spec space.')
        self.stem_models = tuple(model for model in define.FEMORAL_STEM_MODELS if define._has_context_value(model))
        self.specs_by_model = {model: tuple(spec for spec in self.stem_specs if spec[0] == model) for model in self.stem_models}
        self.stem_model_to_id = {model: i for i, model in enumerate(define.FEMORAL_STEM_MODELS)}
        self.stem_spec_to_id = {spec: i for i, spec in enumerate(define.FEMORAL_STEM_SPECS)}
        self.numeric_class_values = numeric_class_values

    def apply(self, batch, seed):
        generator = torch.Generator().manual_seed(int(seed))
        batch_size = int(batch['image'].shape[0])
        stem_model_ids, stem_spec_ids = [], []
        numerics, masks = [], []
        numeric_class_ids, numeric_class_masks = [], []
        for _ in range(batch_size):
            if float(torch.rand((), generator=generator).item()) < 0.5:
                model_index = int(torch.randint(len(self.stem_models), (1,), generator=generator).item())
                stem_model = self.stem_models[model_index]
                model_specs = self.specs_by_model[stem_model]
                spec_index = int(torch.randint(len(model_specs), (1,), generator=generator).item())
                stem_model, stem_size = model_specs[spec_index]
            else:
                spec_index = int(torch.randint(len(self.stem_specs), (1,), generator=generator).item())
                stem_model, stem_size = self.stem_specs[spec_index]
            stem_model_ids.append(self.stem_model_to_id[stem_model])
            stem_spec_ids.append(self.stem_spec_to_id[(stem_model, stem_size)])

            scaled_values = []
            class_ids = []
            for name in NUMERIC_LABEL_NAMES:
                values = self.numeric_class_values[name]
                value_index = int(torch.randint(len(values), (1,), generator=generator).item())
                value = values[value_index]
                min_value, max_value = define.PROSTHESIS_NUMERIC_RANGES[name]
                scaled, _ = define._scale_context_number(value, min_value, max_value)
                scaled_values.append(scaled)
                class_ids.append(value_index)
            numerics.append(scaled_values)
            numeric_class_ids.append(class_ids)
            masks.append([1.0] * len(LABEL_NAMES))
            numeric_class_masks.append([1.0] * len(NUMERIC_LABEL_NAMES))

        batch = dict(batch)
        batch['stem_model_id'] = torch.tensor(stem_model_ids, dtype=torch.long)
        batch['stem_spec_id'] = torch.tensor(stem_spec_ids, dtype=torch.long)
        batch['numerics'] = torch.tensor(numerics, dtype=torch.float32)
        batch['masks'] = torch.tensor(masks, dtype=torch.float32)
        batch['numeric_class_ids'] = torch.tensor(numeric_class_ids, dtype=torch.long)
        batch['numeric_class_masks'] = torch.tensor(numeric_class_masks, dtype=torch.float32)
        return batch


class MetalGeometryCls(torch.nn.Module):
    def __init__(self, numeric_class_counts):
        super().__init__()
        channels = (8, 32, 64, 128, 192, 256)
        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.extend([
                torch.nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                torch.nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
                torch.nn.SiLU(),
            ])
        self.backbone = torch.nn.Sequential(*blocks)
        self.pool = torch.nn.AdaptiveAvgPool3d(1)
        self.neck = torch.nn.Sequential(
            torch.nn.Linear(channels[-1], 256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
        )
        self.stem_model = torch.nn.Linear(256, len(define.FEMORAL_STEM_MODELS))
        self.stem_spec = torch.nn.Linear(256, len(define.FEMORAL_STEM_SPECS))
        self.numeric_heads = torch.nn.ModuleDict({name: torch.nn.Linear(256, count) for name, count in numeric_class_counts.items()})

    def forward(self, image):
        feat = self.backbone(image)
        feat = self.pool(feat).flatten(1)
        feat = self.neck(feat)
        return {
            'stem_model': self.stem_model(feat),
            'stem_spec': self.stem_spec(feat),
            **{name: head(feat) for name, head in self.numeric_heads.items()},
        }


class RFlowMetalGenerator:
    def __init__(self, checkpoint_path, steps, use_amp):
        self.steps = steps
        self.use_amp = use_amp and device.type == 'cuda'
        self.rflow = define.rflow_unet(context_embedding_size=256).to(device)
        self.condition_encoder = define.StructuredProsthesisConditionEncoder(embed_dim=256).to(device)
        ckpt = torch.load(Path(checkpoint_path).resolve(), map_location=device, weights_only=False)
        self.rflow.load_state_dict(ckpt.get('rflow_state_ema', ckpt['rflow_state']))
        self.condition_encoder.load_state_dict(ckpt.get('condition_encoder_state_ema', ckpt['condition_encoder_state']))
        self.rflow.eval()
        self.condition_encoder.eval()
        self.scheduler = define.scheduler_rflow()

    def amp_context(self):
        return autocast(device_type=device.type, enabled=self.use_amp) if self.use_amp else nullcontext()

    @torch.no_grad()
    def generate(self, batch, seed):
        image_shape = tuple(batch['image'].shape)
        bone_latent = torch.zeros(image_shape[0], 4, *image_shape[2:], device=device, dtype=torch.float32)
        stem_model_id = batch['stem_model_id'].to(device)
        stem_spec_id = batch['stem_spec_id'].to(device)
        numerics = batch['numerics'].to(device)
        masks = batch['masks'].to(device)
        bone_latent, masks = define.apply_prosthesis_condition_mode(bone_latent, masks, 'prosthesis_full_only')
        condition_tokens = self.condition_encoder(stem_model_id, stem_spec_id, numerics, masks)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        generated = torch.randn(image_shape, device=device, generator=generator)
        self.scheduler.set_timesteps(num_inference_steps=self.steps)
        timesteps = self.scheduler.timesteps.to(device)
        next_timesteps = torch.cat([timesteps[1:], torch.zeros(1, dtype=timesteps.dtype, device=device)])
        for t, next_t in zip(timesteps, next_timesteps):
            with self.amp_context():
                model_input = torch.cat([generated, bone_latent], dim=1)
                timestep = t.expand(generated.shape[0]).to(device)
                velocity = self.rflow(model_input, timestep, context=condition_tokens)
            generated, _ = self.scheduler.step(velocity, t, generated, next_t)
        return generated.detach()


def masked_cross_entropy(logits, targets, masks):
    losses = torch.nn.functional.cross_entropy(logits.float(), targets, reduction='none')
    losses = losses * masks
    return losses.sum(), masks.sum()


def masked_stem_spec_cross_entropy(logits, targets, stem_model_ids, masks):
    model_spec_mask = build_stem_spec_model_mask(logits.device)
    allowed = model_spec_mask[stem_model_ids]
    masked_logits = logits.float().masked_fill(~allowed, -1e9)
    losses = torch.nn.functional.cross_entropy(masked_logits, targets, reduction='none')
    losses = losses * masks
    return losses.sum(), masks.sum()


def ordered_soft_cross_entropy(logits, target_ids, masks, class_values, sigma):
    values = torch.as_tensor(class_values, device=logits.device, dtype=torch.float32)
    target_values = values[target_ids].unsqueeze(1)
    if sigma > 0:
        soft_targets = torch.exp(-0.5 * ((values.unsqueeze(0) - target_values) / sigma) ** 2)
        soft_targets = soft_targets / torch.clamp(soft_targets.sum(dim=1, keepdim=True), min=1e-12)
    else:
        soft_targets = torch.nn.functional.one_hot(target_ids, num_classes=len(class_values)).float()
    losses = -(soft_targets * torch.nn.functional.log_softmax(logits.float(), dim=1)).sum(dim=1)
    losses = losses * masks
    return losses.sum(), masks.sum()


def compute_losses(outputs, batch, numeric_class_values, numeric_sigmas):
    masks = batch['masks'].to(device)
    stem_model_ids = batch['stem_model_id'].to(device)
    stem_spec_ids = batch['stem_spec_id'].to(device)
    numeric_ids = batch['numeric_class_ids'].to(device)
    numeric_masks = batch['numeric_class_masks'].to(device)

    group_losses = []
    valid_sum = torch.zeros((), device=device)

    model_loss, model_valid = masked_cross_entropy(outputs['stem_model'], stem_model_ids, masks[:, 0])
    if model_valid > 0:
        group_losses.append(model_loss / model_valid)
    valid_sum = valid_sum + model_valid

    spec_loss, spec_valid = masked_stem_spec_cross_entropy(outputs['stem_spec'], stem_spec_ids, stem_model_ids, masks[:, 1])
    if spec_valid > 0:
        group_losses.append(spec_loss / spec_valid)
    valid_sum = valid_sum + spec_valid

    numeric_loss_sum = torch.zeros((), device=device)
    numeric_valid_sum = torch.zeros((), device=device)
    for i, name in enumerate(NUMERIC_LABEL_NAMES):
        loss, valid = ordered_soft_cross_entropy(
            outputs[name],
            numeric_ids[:, i],
            numeric_masks[:, i],
            numeric_class_values[name],
            numeric_sigmas[name],
        )
        numeric_loss_sum = numeric_loss_sum + loss
        numeric_valid_sum = numeric_valid_sum + valid
    if numeric_valid_sum > 0:
        group_losses.append(numeric_loss_sum / numeric_valid_sum)
    valid_sum = valid_sum + numeric_valid_sum

    if group_losses:
        return torch.stack(group_losses).mean(), valid_sum.detach()
    return torch.zeros((), device=device, requires_grad=True), valid_sum.detach()


@torch.no_grad()
def update_metrics(metrics, outputs, batch):
    masks = batch['masks'].to(device)
    numeric_masks = batch['numeric_class_masks'].to(device)
    stem_model_ids = batch['stem_model_id'].to(device)
    stem_spec_ids = batch['stem_spec_id'].to(device)
    numeric_ids = batch['numeric_class_ids'].to(device)

    head_targets = {
        'stem_model': (stem_model_ids, masks[:, 0]),
        'cup_outer': (numeric_ids[:, 0], numeric_masks[:, 0]),
        'head_outer': (numeric_ids[:, 1], numeric_masks[:, 1]),
        'head_offset': (numeric_ids[:, 2], numeric_masks[:, 2]),
        'liner_offset': (numeric_ids[:, 3], numeric_masks[:, 3]),
    }
    exact_correct = torch.ones(stem_model_ids.shape[0], device=device, dtype=torch.bool)
    exact_valid = torch.zeros(stem_model_ids.shape[0], device=device, dtype=torch.bool)
    for name, (target, mask) in head_targets.items():
        pred = outputs[name].argmax(dim=1)
        valid = mask > 0
        correct = (pred == target) & valid
        metrics[f'{name}_correct'] += correct.float().sum().item()
        metrics[f'{name}_valid'] += valid.float().sum().item()
        exact_correct &= (~valid) | correct
        exact_valid |= valid

    spec_valid = masks[:, 1] > 0
    spec_global_pred = outputs['stem_spec'].argmax(dim=1)
    spec_global_correct = (spec_global_pred == stem_spec_ids) & spec_valid
    metrics['stem_spec_global_correct'] += spec_global_correct.float().sum().item()
    metrics['stem_spec_global_valid'] += spec_valid.float().sum().item()

    model_spec_mask = build_stem_spec_model_mask(outputs['stem_spec'].device)
    allowed = model_spec_mask[stem_model_ids]
    spec_within_logits = outputs['stem_spec'].float().masked_fill(~allowed, -1e9)
    spec_within_pred = spec_within_logits.argmax(dim=1)
    spec_within_correct = (spec_within_pred == stem_spec_ids) & spec_valid
    metrics['stem_spec_correct'] += spec_within_correct.float().sum().item()
    metrics['stem_spec_valid'] += spec_valid.float().sum().item()
    exact_correct &= (~spec_valid) | spec_within_correct
    exact_valid |= spec_valid

    metrics['all_labeled_exact_correct'] += (exact_correct & exact_valid).float().sum().item()
    metrics['all_labeled_exact_valid'] += exact_valid.float().sum().item()


def metric_value(metrics, name):
    valid = metrics.get(f'{name}_valid', 0.0)
    return metrics.get(f'{name}_correct', 0.0) / valid if valid > 0 else float('nan')


def balanced_score(metrics):
    stem_model_score = metric_value(metrics, 'stem_model')
    stem_spec_score = metric_value(metrics, 'stem_spec')
    numeric_scores = [metric_value(metrics, name) for name in NUMERIC_LABEL_NAMES]
    numeric_scores = [score for score in numeric_scores if np.isfinite(score)]
    numeric_score = float(np.mean(numeric_scores)) if numeric_scores else float('nan')
    scores = [stem_model_score, stem_spec_score, numeric_score]
    scores = [score for score in scores if np.isfinite(score)]
    return float(np.mean(scores)) if scores else float('nan')


def format_metric_summary(domain, loss, metrics):
    fields = {
        'score': balanced_score(metrics),
        'model': metric_value(metrics, 'stem_model'),
        'spec': metric_value(metrics, 'stem_spec'),
        'spec_g': metric_value(metrics, 'stem_spec_global'),
        'cup': metric_value(metrics, 'cup_outer'),
        'head': metric_value(metrics, 'head_outer'),
        'hoff': metric_value(metrics, 'head_offset'),
        'liner': metric_value(metrics, 'liner_offset'),
        'exact': metric_value(metrics, 'all_labeled_exact'),
    }
    values = ' '.join(f'{name}={value:.4f}' for name, value in fields.items())
    return f'{domain}: loss={loss:.4f} {values}'


def repeat_batch(batch, repeats):
    if repeats <= 1:
        return batch
    repeated = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            repeated[key] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
        else:
            repeated[key] = value
    return repeated


def log_line(*args):
    print(*args, flush=True)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    generator,
    extended_sampler,
    domain,
    numeric_class_values,
    numeric_sigmas,
    epoch,
    writer,
    train,
    progress_enabled,
    max_steps=None,
    synthetic_batch_size=1,
    probe_loader=None,
    step_offset=0,
):
    model.train(train)
    metrics = {f'{name}_{kind}': 0.0 for name in METRIC_NAMES for kind in ('correct', 'valid')}
    loss_total = 0.0
    steps = 0
    main_steps = len(loader) if max_steps is None else min(max_steps, len(loader))
    probe_steps = len(probe_loader) if probe_loader is not None else 0
    main_iter = loader if max_steps is None else itertools.islice(loader, main_steps)
    loader_iter = itertools.chain(probe_loader, main_iter) if probe_loader is not None else main_iter
    epoch_steps = probe_steps + main_steps
    pbar = tqdm(loader_iter, total=epoch_steps, desc=f'{domain} ' + ('train' if train else 'val'), disable=not progress_enabled)
    for step, batch in enumerate(pbar):
        sample_seed = step_offset + step if train else epoch * epoch_steps + step
        if domain == 'extended':
            if extended_sampler is None:
                raise RuntimeError('Extended-domain training requires an extended label sampler.')
            batch = extended_sampler.apply(batch, seed=sample_seed)
        if domain in {'generated', 'extended'}:
            batch = repeat_batch(batch, synthetic_batch_size)
        if domain in {'generated', 'extended'}:
            if generator is None:
                raise RuntimeError('Generated-domain training requires checkpoints/rflow_last.pt under train.root.')
            image = generator.generate(batch, seed=sample_seed)
        else:
            image = batch['image'].to(device, non_blocking=True)

        context = autocast(device_type=device.type, enabled=(scaler is not None)) if scaler is not None else nullcontext()
        with torch.set_grad_enabled(train), context:
            outputs = model(image)
            loss, _ = compute_losses(outputs, batch, numeric_class_values, numeric_sigmas)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        update_metrics(metrics, outputs, batch)
        loss_total += loss.item()
        steps += 1
        if train:
            writer.add_scalar(f'loss_step/train/{domain}', loss.item(), step_offset + step)
        if progress_enabled:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'stem': f'{metric_value(metrics, "stem_model"):.3f}'})

    split = 'train' if train else 'val'
    writer.add_scalar(f'loss/{split}/{domain}', loss_total / max(steps, 1), epoch)
    for name in METRIC_NAMES:
        writer.add_scalar(f'accuracy/{split}/{domain}/{name}', metric_value(metrics, name), epoch)
    writer.add_scalar(f'score/{split}/{domain}/balanced', balanced_score(metrics), epoch)
    return loss_total / max(steps, 1), metrics


def main():
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--domain', choices=('real', 'generated', 'extended'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()
    train_cfg = cfg['train']['metal_cls']
    train_root = Path(str(cfg['train']['root']))
    ckpt_dir = train_root / 'checkpoints'
    log_root = train_root / 'logs'
    progress_enabled = sys.stderr.isatty()
    use_amp = bool(train_cfg['use_amp'] and device.type == 'cuda')

    train_files, val_files, _, _ = build_split_files(cfg)
    if not train_files or not val_files:
        raise RuntimeError('Empty train or validation split.')
    numeric_class_values = build_numeric_bins()
    numeric_sigmas = {name: float(train_cfg['numeric_soft_label_sigma'][name]) for name in NUMERIC_LABEL_NAMES}
    synthetic_batch_size = int(train_cfg.get('synthetic_batch_size', 1))
    if synthetic_batch_size < 1:
        raise ValueError('synthetic_batch_size must be >= 1')

    image_sf, image_mean = load_vae_stats(ckpt_dir, 'metal')
    cond_sf, cond_mean = load_vae_stats(ckpt_dir, 'pre')
    transforms = Compose(cls_transforms(image_mean, image_sf, cond_mean, cond_sf, numeric_class_values))
    train_ds = Dataset(data=train_files, transform=transforms)
    val_ds = Dataset(data=val_files, transform=transforms)
    probe_largest_count = int(train_cfg.get('probe_largest_samples', 0))
    probe_files = select_largest_latent_files(train_files, probe_largest_count)
    probe_ds = Dataset(data=probe_files, transform=transforms)
    num_workers = int(train_cfg['num_workers'])
    loader_kwargs = {
        'batch_size': 1,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': num_workers > 0,
    }
    train_sampler = build_balanced_train_sampler(train_files, numeric_class_values)
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    probe_loader = DataLoader(probe_ds, shuffle=False, **loader_kwargs) if probe_files else None
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = MetalGeometryCls({name: len(numeric_class_values[name]) for name in NUMERIC_LABEL_NAMES}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['lr']), weight_decay=float(train_cfg['weight_decay']))
    scaler = GradScaler() if use_amp else None

    generator = None
    generator_ckpt = ckpt_dir / 'rflow_last.pt'
    if generator_ckpt.exists():
        try:
            generator = RFlowMetalGenerator(generator_ckpt, int(train_cfg['generated_steps']), use_amp=use_amp)
        except Exception as exc:
            if args.domain in {'generated', 'extended'}:
                raise
            log_line(f'Warning: failed to load {generator_ckpt.as_posix()}; synthetic validation is disabled. {exc}')
    elif args.domain in {'generated', 'extended'}:
        raise RuntimeError(f'RFlow checkpoint does not exist: {generator_ckpt.as_posix()}')
    extended_sampler = ExtendedLabelSampler(numeric_class_values)

    task_name = f'metal_cls_{args.domain}'
    checkpoint_path = ckpt_dir / f'{task_name}_last.pt'
    best_path = ckpt_dir / f'{task_name}_best.pt'
    start_epoch = 0
    best_score = -1.0
    if args.resume or bool(train_cfg.get('resume', False)):
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        if use_amp and 'scaler' in loaded:
            scaler.load_state_dict(loaded['scaler'])
        start_epoch = int(loaded['epoch']) + 1
        best_score = float(loaded['best_score'])

    suffix = datetime.now().strftime(f'{task_name}_%Y%m%d_%H%M%S')
    log_dir = log_root / suffix
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    log_line('Domain:\t', args.domain)
    log_line('Train condition pool:\t', len(train_files))
    log_line('Val:\t', len(val_files))
    log_line('Train sampler:\t balanced by mean(stem_model, stem_spec, numeric-parameter group)')
    log_line('Loss rule:\t mean(stem_model, stem_spec within true stem_model, numeric-parameter group)')
    log_line('Numeric class values:\t', numeric_class_values)
    if args.domain in {'generated', 'extended'}:
        synthetic_train_conditions = int(train_cfg.get('synthetic_train_samples_per_epoch', 0))
        synthetic_train_text = synthetic_train_conditions if synthetic_train_conditions > 0 else 'all'
        log_line('Synthetic train conditions per epoch:\t', synthetic_train_text)
        log_line('Synthetic sample batch size:\t', synthetic_batch_size)
        if synthetic_train_conditions > 0:
            log_line('Synthetic train samples per epoch:\t', synthetic_train_conditions * synthetic_batch_size)
        log_line('Epoch-0 largest probe conditions:\t', len(probe_files))
    if generator is not None:
        synthetic_val_conditions = int(train_cfg.get('synthetic_val_samples', 0))
        synthetic_val_text = synthetic_val_conditions if synthetic_val_conditions > 0 else 'all'
        log_line('Synthetic validation conditions:\t', synthetic_val_text)

    for epoch in range(start_epoch, int(train_cfg['num_epochs'])):
        train_max_steps = None
        if args.domain in {'generated', 'extended'}:
            synthetic_train_limit = int(train_cfg.get('synthetic_train_samples_per_epoch', 0))
            train_max_steps = synthetic_train_limit if synthetic_train_limit > 0 else None
        train_main_steps = len(train_loader) if train_max_steps is None else min(train_max_steps, len(train_loader))
        train_probe_steps = len(probe_loader) if args.domain in {'generated', 'extended'} and probe_loader is not None else 0
        train_step_offset = epoch * train_main_steps
        if args.domain in {'generated', 'extended'} and epoch > 0:
            train_step_offset += train_probe_steps
        log_line(
            f'Epoch {epoch}: train {args.domain} start '
            f'conditions={train_main_steps + (train_probe_steps if epoch == 0 else 0)} '
            f'synthetic_batch={synthetic_batch_size if args.domain in {"generated", "extended"} else 1}'
        )
        train_loss, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            generator,
            extended_sampler,
            args.domain,
            numeric_class_values,
            numeric_sigmas,
            epoch,
            writer,
            True,
            progress_enabled,
            max_steps=train_max_steps,
            synthetic_batch_size=synthetic_batch_size,
            probe_loader=probe_loader if args.domain in {'generated', 'extended'} and epoch == 0 else None,
            step_offset=train_step_offset,
        )
        log_line(f'Epoch {epoch}: train {args.domain} done loss={train_loss:.4f}')
        if epoch % int(train_cfg['val_interval']) == 0:
            log_line(f'Epoch {epoch}: validation real start')
            val_real_loss, val_real_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                scaler,
                None,
                extended_sampler,
                'real',
                numeric_class_values,
                numeric_sigmas,
                epoch,
                writer,
                False,
                progress_enabled,
            )
            val_generated_loss = None
            val_generated_metrics = None
            if generator is not None:
                synthetic_val_limit = int(train_cfg.get('synthetic_val_samples', 0))
                log_line(f'Epoch {epoch}: validation generated start')
                val_generated_loss, val_generated_metrics = run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    scaler,
                    generator,
                    extended_sampler,
                    'generated',
                    numeric_class_values,
                    numeric_sigmas,
                    epoch,
                    writer,
                    False,
                    progress_enabled,
                    max_steps=synthetic_val_limit if synthetic_val_limit > 0 else None,
                    synthetic_batch_size=synthetic_batch_size,
                )
            val_extended_loss = None
            val_extended_metrics = None
            if generator is not None:
                synthetic_val_limit = int(train_cfg.get('synthetic_val_samples', 0))
                log_line(f'Epoch {epoch}: validation extended start')
                val_extended_loss, val_extended_metrics = run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    scaler,
                    generator,
                    extended_sampler,
                    'extended',
                    numeric_class_values,
                    numeric_sigmas,
                    epoch,
                    writer,
                    False,
                    progress_enabled,
                    max_steps=synthetic_val_limit if synthetic_val_limit > 0 else None,
                    synthetic_batch_size=synthetic_batch_size,
                )
            primary_metrics = val_real_metrics
            if args.domain == 'generated' and val_generated_metrics is not None:
                primary_metrics = val_generated_metrics
            elif args.domain == 'extended' and val_extended_metrics is not None:
                primary_metrics = val_extended_metrics
            score = balanced_score(primary_metrics)
            summaries = [format_metric_summary('real', val_real_loss, val_real_metrics)]
            if val_generated_metrics is not None:
                summaries.append(format_metric_summary('generated', val_generated_loss, val_generated_metrics))
            if val_extended_metrics is not None:
                summaries.append(format_metric_summary('extended', val_extended_loss, val_extended_metrics))
            log_line(
                f'Epoch {epoch}: train_loss={train_loss:.4f} best_score={max(best_score, score):.4f} current_score={score:.4f} | '
                + ' | '.join(summaries)
            )
            ckpt = {
                'epoch': epoch,
                'domain': args.domain,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'numeric_class_values': numeric_class_values,
                'numeric_sigmas': numeric_sigmas,
                'synthetic_batch_size': synthetic_batch_size,
                'probe_largest_samples': len(probe_files),
                'val_real_metrics': val_real_metrics,
                'val_generated_metrics': val_generated_metrics,
                'val_extended_metrics': val_extended_metrics,
                'loss_rule': 'mean(stem_model_loss, stem_spec_within_true_model_loss, mean(cup_outer_loss, head_outer_loss, head_offset_loss, liner_offset_loss))',
                'score_rule': 'mean(stem_model_acc, stem_spec_within_true_model_acc, mean(cup_outer_acc, head_outer_acc, head_offset_acc, liner_offset_acc))',
                'best_score': max(best_score, score),
            }
            if use_amp:
                ckpt['scaler'] = scaler.state_dict()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, checkpoint_path)
            if score > best_score:
                best_score = score
                torch.save(ckpt, best_path)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    writer.close()


if __name__ == '__main__':
    main()
