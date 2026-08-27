"""Compare flow-matching residual scoring (method 1) against the metal classifier.

Method 1 scores a metal latent y under candidate catalog conditions c with bone
zeroed: L_FM(c) = E_{t,ε} || v_θ(y_t, t, c) - (y - ε) ||^2, then picks argmin_c.
Search is hierarchical: 19 stem models (spec/numerics dropped), then specs of
the predicted model. The classifier is evaluated on the same y tensors.
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tomlkit
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from tqdm import tqdm

import c6_train_metal_cls as metal_cls_mod
import define

STEM_MODELS = tuple(model for model in define.FEMORAL_STEM_MODELS if define._has_context_value(model))
STEM_MODEL_IDS = tuple(define.FEMORAL_STEM_MODELS.index(model) for model in STEM_MODELS)
STEM_SPECS_BY_MODEL = {
    model: tuple(
        (define.FEMORAL_STEM_SPECS.index((model, size)), size)
        for size in define.FEMORAL[model]
        if define._has_context_value(size)
    )
    for model in STEM_MODELS
}


def log_line(*args):
    print(*args, flush=True)


def amp_context(enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type='cuda', enabled=True)


def load_vae_stats(path):
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    return loaded['scale_factor'], loaded['global_mean']


def candidate_condition(model_ids, spec_ids, device):
    batch = len(model_ids)
    stem_model_id = torch.as_tensor(model_ids, device=device, dtype=torch.long)
    stem_spec_id = torch.as_tensor(spec_ids, device=device, dtype=torch.long)
    numerics = torch.zeros(batch, 4, device=device, dtype=torch.float32)
    masks = torch.zeros(batch, 6, device=device, dtype=torch.float32)
    masks[:, 0] = 1.0
    if any(int(spec_id) >= 0 for spec_id in spec_ids):
        masks[:, 1] = 1.0
        stem_spec_id = stem_spec_id.clamp(min=0)
    else:
        stem_spec_id = torch.zeros(batch, device=device, dtype=torch.long)
    return stem_model_id, stem_spec_id, numerics, masks


@torch.no_grad()
def make_score_draws(scheduler, image, draws, generator):
    pairs = []
    for _ in range(draws):
        timesteps = scheduler.sample_timesteps(image)
        noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        noisy = scheduler.add_noise(original_samples=image, noise=noise, timesteps=timesteps)
        pairs.append((timesteps, noisy, image - noise))
    return pairs


@torch.no_grad()
def flow_match_scores(rflow, condition_encoder, image, candidates, score_draws, candidate_batch, use_amp):
    device = image.device
    n_cand = int(candidates['stem_model_id'].shape[0])
    scores = torch.zeros(n_cand, device=device, dtype=torch.float32)
    bone = torch.zeros(1, 4, *image.shape[2:], device=device, dtype=image.dtype)

    for timesteps, noisy, target in score_draws:
        for start in range(0, n_cand, candidate_batch):
            stop = min(start + candidate_batch, n_cand)
            width = stop - start
            tokens = condition_encoder(
                candidates['stem_model_id'][start:stop],
                candidates['stem_spec_id'][start:stop],
                candidates['numerics'][start:stop],
                candidates['masks'][start:stop],
            )
            with amp_context(use_amp):
                velocity = rflow(
                    x=torch.cat([noisy.expand(width, -1, -1, -1, -1), bone.expand(width, -1, -1, -1, -1)], dim=1),
                    timesteps=timesteps.expand(width),
                    context=tokens,
                )
            residual = (velocity.float() - target.float()).flatten(1).pow(2).mean(dim=1)
            scores[start:stop] += residual
    return scores / float(len(score_draws))


def pack_candidates(model_ids, spec_ids, device):
    stem_model_id, stem_spec_id, numerics, masks = candidate_condition(model_ids, spec_ids, device)
    return {
        'stem_model_id': stem_model_id,
        'stem_spec_id': stem_spec_id,
        'numerics': numerics,
        'masks': masks,
    }


@torch.no_grad()
def generate_metal(rflow, condition_encoder, scheduler, image_shape, bone, tokens, steps, seed, use_amp):
    device = bone.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    generated = torch.randn(image_shape, device=device, generator=generator)
    scheduler.set_timesteps(num_inference_steps=steps)
    timesteps = scheduler.timesteps.to(device)
    next_timesteps = torch.cat([timesteps[1:], torch.zeros(1, dtype=timesteps.dtype, device=device)])
    for t, next_t in zip(timesteps, next_timesteps):
        with amp_context(use_amp):
            velocity = rflow(torch.cat([generated, bone], dim=1), t.expand(generated.shape[0]).to(device), context=tokens)
        generated, _ = scheduler.step(velocity, t, generated, next_t)
    return generated.detach()


def classify_outputs(metal_cls, image, use_amp):
    with amp_context(use_amp):
        return metal_cls(image)


def spec_pred_from_logits(spec_logits, model_id):
    allowed = metal_cls_mod.build_stem_spec_model_mask(spec_logits.device)[int(model_id)]
    if bool(allowed.any()):
        spec_logits = spec_logits.float().masked_fill(~allowed.unsqueeze(0), -1e9)
    return int(spec_logits.argmax(dim=1)[0])


def summarize(rows, key_prefix):
    model_labeled = [row for row in rows if row['model_labeled']]
    spec_labeled = [row for row in rows if row['spec_labeled']]
    n_model = len(model_labeled)
    n_spec = len(spec_labeled)
    model_top1 = sum(row[f'{key_prefix}_model_correct'] for row in model_labeled)
    model_top3 = sum(row[f'{key_prefix}_model_top3'] for row in model_labeled)
    spec_true = sum(row[f'{key_prefix}_spec_within_true_correct'] for row in spec_labeled)
    spec_pred = sum(row[f'{key_prefix}_spec_within_pred_correct'] for row in spec_labeled)
    return {
        'n_model': n_model,
        'n_spec': n_spec,
        'model_top1': model_top1 / n_model if n_model else float('nan'),
        'model_top3': model_top3 / n_model if n_model else float('nan'),
        'spec_within_true': spec_true / n_spec if n_spec else float('nan'),
        'spec_within_pred': spec_pred / n_spec if n_spec else float('nan'),
        'model_top1_count': model_top1,
        'model_top3_count': model_top3,
        'spec_within_true_count': spec_true,
        'spec_within_pred_count': spec_pred,
    }


def format_summary(name, stats):
    return (
        f'{name}: model={stats["model_top1"]:.4f} ({stats["model_top1_count"]}/{stats["n_model"]}) '
        f'top3={stats["model_top3"]:.4f} ({stats["model_top3_count"]}/{stats["n_model"]}) '
        f'spec_true={stats["spec_within_true"]:.4f} ({stats["spec_within_true_count"]}/{stats["n_spec"]}) '
        f'spec_pred={stats["spec_within_pred"]:.4f} ({stats["spec_within_pred_count"]}/{stats["n_spec"]})'
    )


def majority_baseline(rows):
    labeled = [row['true_model'] for row in rows if row['model_labeled']]
    if not labeled:
        return float('nan'), 0
    counts = {}
    for name in labeled:
        counts[name] = counts.get(name, 0) + 1
    majority = max(counts.values())
    return majority / len(labeled), majority


def evaluate_source(
    source,
    loader,
    rflow,
    condition_encoder,
    scheduler,
    metal_cls,
    device,
    draws,
    candidate_batch,
    sample_steps,
    use_amp,
    seed,
):
    model_candidates = pack_candidates(STEM_MODEL_IDS, [-1] * len(STEM_MODEL_IDS), device)
    rows = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    pbar = tqdm(loader, desc=f'score {source}', disable=not sys.stderr.isatty())
    for step, batch in enumerate(pbar):
        image = batch['image'].to(device, non_blocking=True)
        bone = batch['condition'].to(device, non_blocking=True)
        true_model_id = int(batch['stem_model_id'][0])
        true_spec_id = int(batch['stem_spec_id'][0])
        masks = batch['masks'][0]
        model_labeled = float(masks[0]) > 0
        spec_labeled = float(masks[1]) > 0
        true_model = define.FEMORAL_STEM_MODELS[true_model_id] if model_labeled else ''
        true_spec = define.FEMORAL_STEM_SPECS[true_spec_id][1] if spec_labeled else ''

        if source == 'real':
            y = image
        else:
            if source == 'generated':
                cond_bone, cond_masks = define.apply_prosthesis_condition_mode(
                    torch.zeros_like(bone),
                    batch['masks'].to(device),
                    'prosthesis_full_only',
                )
                tokens = condition_encoder(
                    batch['stem_model_id'].to(device),
                    batch['stem_spec_id'].to(device),
                    batch['numerics'].to(device),
                    cond_masks,
                )
            elif source == 'bone_only':
                cond_bone, cond_masks = define.apply_prosthesis_condition_mode(bone, batch['masks'].to(device), 'bone_only')
                tokens = condition_encoder(
                    batch['stem_model_id'].to(device),
                    batch['stem_spec_id'].to(device),
                    batch['numerics'].to(device),
                    cond_masks,
                )
            else:
                raise ValueError(f'Unknown source: {source}')
            y = generate_metal(
                rflow,
                condition_encoder,
                scheduler,
                image.shape,
                cond_bone,
                tokens,
                sample_steps,
                seed + step,
                use_amp,
            )

        cls_out = classify_outputs(metal_cls, y, use_amp)
        cls_model_id = int(cls_out['stem_model'].argmax(dim=1)[0])
        cls_model_top3 = cls_out['stem_model'][0].topk(min(3, cls_out['stem_model'].shape[1])).indices.tolist()
        cls_spec_true = spec_pred_from_logits(cls_out['stem_spec'], true_model_id) if model_labeled else -1
        cls_spec_pred = spec_pred_from_logits(cls_out['stem_spec'], cls_model_id)

        score_draws = make_score_draws(scheduler, y, draws, generator)
        model_scores = flow_match_scores(
            rflow,
            condition_encoder,
            y,
            model_candidates,
            score_draws,
            candidate_batch,
            use_amp,
        )
        model_order = torch.argsort(model_scores)
        fm_model_index = int(model_order[0])
        fm_model_id = STEM_MODEL_IDS[fm_model_index]
        fm_model_top3 = [STEM_MODEL_IDS[int(index)] for index in model_order[:3]]
        fm_true_rank = None
        if model_labeled and true_model_id in STEM_MODEL_IDS:
            fm_true_rank = int((model_order == STEM_MODEL_IDS.index(true_model_id)).nonzero(as_tuple=False)[0]) + 1

        spec_cache = {}

        def score_specs(model_name):
            if model_name in spec_cache:
                return spec_cache[model_name]
            spec_entries = STEM_SPECS_BY_MODEL.get(model_name, ())
            if not spec_entries:
                spec_cache[model_name] = (-1, None)
                return spec_cache[model_name]
            spec_ids = [item[0] for item in spec_entries]
            model_ids = [define.FEMORAL_STEM_MODELS.index(model_name)] * len(spec_ids)
            spec_scores = flow_match_scores(
                rflow,
                condition_encoder,
                y,
                pack_candidates(model_ids, spec_ids, device),
                score_draws,
                candidate_batch,
                use_amp,
            )
            best = int(spec_scores.argmin())
            spec_cache[model_name] = (spec_ids[best], float(spec_scores[best].item()))
            return spec_cache[model_name]

        fm_spec_pred, _ = score_specs(define.FEMORAL_STEM_MODELS[fm_model_id])
        fm_spec_true, _ = score_specs(true_model) if model_labeled else (-1, None)

        row = {
            'prl': batch['prl'][0] if isinstance(batch['prl'], (list, tuple)) else str(batch['prl'][0]),
            'source': source,
            'model_labeled': model_labeled,
            'spec_labeled': spec_labeled,
            'true_model': true_model,
            'true_spec': true_spec,
            'true_model_id': true_model_id,
            'true_spec_id': true_spec_id,
            'cls_model': define.FEMORAL_STEM_MODELS[cls_model_id],
            'cls_model_correct': bool(model_labeled and cls_model_id == true_model_id),
            'cls_model_top3': bool(model_labeled and true_model_id in cls_model_top3),
            'cls_spec_within_true_correct': bool(spec_labeled and cls_spec_true == true_spec_id),
            'cls_spec_within_pred_correct': bool(spec_labeled and cls_spec_pred == true_spec_id),
            'fm_model': define.FEMORAL_STEM_MODELS[fm_model_id],
            'fm_model_correct': bool(model_labeled and fm_model_id == true_model_id),
            'fm_model_top3': bool(model_labeled and true_model_id in fm_model_top3),
            'fm_true_rank': fm_true_rank,
            'fm_true_score': float(model_scores[STEM_MODEL_IDS.index(true_model_id)].item())
            if model_labeled and true_model_id in STEM_MODEL_IDS
            else None,
            'fm_best_score': float(model_scores[fm_model_index].item()),
            'fm_spec_within_true_correct': bool(spec_labeled and fm_spec_true == true_spec_id),
            'fm_spec_within_pred_correct': bool(spec_labeled and fm_spec_pred == true_spec_id),
        }
        rows.append(row)
        if model_labeled:
            pbar.set_postfix({
                'fm': f'{summarize(rows, "fm")["model_top1"]:.3f}',
                'cls': f'{summarize(rows, "cls")["model_top1"]:.3f}',
            })
    return rows


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml')
    parser.add_argument('--rflow', default='release/rflow_last.pt')
    parser.add_argument('--cls', default='release/metal_cls_generated_best.pt')
    parser.add_argument('--vae-pre', default='release/vae_pre_best.pt')
    parser.add_argument('--vae-metal', default='release/vae_metal_best.pt')
    parser.add_argument('--sources', default='real,generated,bone_only')
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--candidate-batch', type=int, default=16)
    parser.add_argument('--sample-steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-samples', type=int, default=0)
    parser.add_argument('--out', default='doc/rflow_score_vs_cls.json')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for method-1 scoring.')
    device = torch.device('cuda')
    use_amp = True

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()
    _, val_files, _, _ = metal_cls_mod.build_split_files(cfg)
    if args.num_samples > 0:
        val_files = val_files[: args.num_samples]
    if not val_files:
        raise RuntimeError('Empty validation split.')

    image_sf, image_mean = load_vae_stats(Path(args.vae_metal))
    cond_sf, cond_mean = load_vae_stats(Path(args.vae_pre))
    transforms = Compose(define.rflow_transforms(image_mean, image_sf, cond_mean, cond_sf))
    loader = DataLoader(Dataset(data=val_files, transform=transforms), batch_size=1, num_workers=0)

    log_line('Device:\t', device)
    log_line('Val:\t', len(val_files))
    log_line('Models:\t', len(STEM_MODELS))
    log_line('Draws:\t', args.draws)
    log_line('Candidate batch:\t', args.candidate_batch)

    rflow = define.rflow_unet(context_embedding_size=256).to(device)
    condition_encoder = define.StructuredProsthesisConditionEncoder(embed_dim=256).to(device)
    rflow_ckpt = torch.load(Path(args.rflow).resolve(), map_location=device, weights_only=False)
    rflow.load_state_dict(rflow_ckpt.get('rflow_state_ema', rflow_ckpt['rflow_state']))
    condition_encoder.load_state_dict(rflow_ckpt.get('condition_encoder_state_ema', rflow_ckpt['condition_encoder_state']))
    rflow.eval()
    condition_encoder.eval()
    log_line('RFlow:\t', Path(args.rflow).resolve(), 'epoch', rflow_ckpt.get('epoch'))

    cls_ckpt = torch.load(Path(args.cls).resolve(), map_location=device, weights_only=False)
    numeric_class_values = cls_ckpt.get('numeric_class_values') or metal_cls_mod.build_numeric_bins()
    numeric_class_values = {name: tuple(numeric_class_values[name]) for name in metal_cls_mod.NUMERIC_LABEL_NAMES}
    metal_cls = metal_cls_mod.MetalGeometryCls({name: len(numeric_class_values[name]) for name in metal_cls_mod.NUMERIC_LABEL_NAMES}).to(device)
    metal_cls.load_state_dict(cls_ckpt['model'])
    metal_cls.eval()
    log_line('Cls:\t', Path(args.cls).resolve(), 'epoch', cls_ckpt.get('epoch'), 'domain', cls_ckpt.get('domain'))

    scheduler = define.scheduler_rflow()
    sources = tuple(item.strip() for item in args.sources.split(',') if item.strip())
    results = {'args': vars(args), 'rflow_epoch': rflow_ckpt.get('epoch'), 'cls_epoch': cls_ckpt.get('epoch'), 'sources': {}}

    for source in sources:
        log_line(f'==== source {source}')
        rows = evaluate_source(
            source,
            loader,
            rflow,
            condition_encoder,
            scheduler,
            metal_cls,
            device,
            args.draws,
            args.candidate_batch,
            args.sample_steps,
            use_amp,
            args.seed,
        )
        baseline, baseline_count = majority_baseline(rows)
        fm_stats = summarize(rows, 'fm')
        cls_stats = summarize(rows, 'cls')
        ranks = [row['fm_true_rank'] for row in rows if row['fm_true_rank'] is not None]
        mean_rank = float(np.mean(ranks)) if ranks else float('nan')
        log_line(format_summary('FM', fm_stats))
        log_line(format_summary('CNN', cls_stats))
        log_line(f'Majority baseline: {baseline:.4f} ({baseline_count}/{fm_stats["n_model"]})')
        log_line(f'FM mean rank of true model: {mean_rank:.2f} / {len(STEM_MODELS)}')
        results['sources'][source] = {
            'fm': fm_stats,
            'cls': cls_stats,
            'majority_baseline': baseline,
            'fm_mean_true_rank': mean_rank,
            'rows': rows,
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    log_line('Wrote:\t', out_path.resolve())


if __name__ == '__main__':
    main()
