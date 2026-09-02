import sys
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import itk
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, DivisiblePadd, EnsureChannelFirstd, Lambdad, LoadImaged, SpatialPadd
from torch import autocast

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import define  # noqa: E402

NUMERIC_LABEL_NAMES = ('cup_outer', 'head_outer', 'head_offset', 'liner_offset')
NUMERIC_CLASS_STEPS = {
    'cup_outer': 2.0,
    'head_outer': 2.0,
    'head_offset': 0.5,
}

CONDITION_MODE_LABELS = {
    'unconditional': '完全无条件',
    'bone_only': '仅骨骼条件',
    'prosthesis_full_only': '仅假体全参数',
    'bone_stem_model': '骨骼 + 柄型号',
    'bone_stem_model_spec': '骨骼 + 柄型号规格',
    'bone_prosthesis_full': '骨骼 + 假体全参数',
}
CONDITION_MODE_PARAM_LEVEL = {
    'unconditional': None,
    'bone_only': None,
    'prosthesis_full_only': 'full',
    'bone_stem_model': 'model',
    'bone_stem_model_spec': 'model_spec',
    'bone_prosthesis_full': 'full',
}

STEM_MODELS = tuple(model for model in define.FEMORAL_STEM_MODELS if define._has_context_value(model))


def specs_for_model(model):
    return tuple(size for size in define.FEMORAL.get(model, []) if define._has_context_value(size))


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


NUMERIC_OPTIONS = build_numeric_bins()
NUMERIC_FALLBACKS = {
    'cup_outer': 50.0,
    'head_outer': 32.0,
    'head_offset': 0.0,
    'liner_offset': 0.0,
}


def build_stem_spec_model_mask(device):
    mask = torch.zeros((len(define.FEMORAL_STEM_MODELS), len(define.FEMORAL_STEM_SPECS)), dtype=torch.bool, device=device)
    model_to_id = {model: i for i, model in enumerate(define.FEMORAL_STEM_MODELS)}
    for spec_id, (model, size) in enumerate(define.FEMORAL_STEM_SPECS):
        if define._has_context_value(model) and define._has_context_value(size):
            mask[model_to_id[model], spec_id] = True
    return mask


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


def _device(cpu=False):
    if cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _optional_float(value):
    if not define._has_context_value(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def nearest_numeric(value, options, fallback=None):
    options = tuple(options)
    if not options:
        return fallback
    if value is None:
        return fallback if fallback in options or fallback is None else options[len(options) // 2]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback if fallback in options else options[0]
    return min(options, key=lambda item: abs(float(item) - value))


def case_prosthesis_values(ctx):
    ctx = ctx or {}
    femoral_spec = define._get_context_sequence(ctx, 'femoral_spec', 2)
    stem_model = femoral_spec[0] if define._has_context_value(femoral_spec[0]) else None
    stem_size = femoral_spec[1] if define._has_context_value(femoral_spec[1]) else None
    return {
        'stem_model': stem_model if stem_model in STEM_MODELS else None,
        'stem_size': stem_size if stem_model in define.FEMORAL and stem_size in specs_for_model(stem_model) else None,
        'cup_outer': _optional_float(define._get_context_value(ctx, 'cup_outer')),
        'head_outer': _optional_float(define._get_context_value(ctx, 'head_outer')),
        'head_offset': _optional_float(define._get_context_value(ctx, 'head_offset')),
        'liner_offset': _optional_float(ctx.get('liner_offset')),
    }


def format_condition_desc(mode=None, stem_model=None, stem_size=None, cup_outer=None, head_outer=None, head_offset=None, liner_offset=None, seed=None):
    parts = []
    mode_label = CONDITION_MODE_LABELS.get(mode, mode)
    if mode_label:
        parts.append(mode_label)
    if define._has_context_value(stem_model):
        parts.append(f'柄型号 {stem_model}')
    if define._has_context_value(stem_size):
        parts.append(f'柄规格 {stem_size}')
    if cup_outer is not None:
        parts.append(f'杯直径 {float(cup_outer):.0f}')
    if head_outer is not None:
        parts.append(f'头直径 {float(head_outer):.0f}')
    if head_offset is not None:
        parts.append(f'头偏距 {float(head_offset):+g}')
    if liner_offset is not None:
        parts.append(f'衬偏心 {float(liner_offset):g}')
    if seed is not None:
        parts.append(f'种子 {seed}')
    return ' '.join(parts)


def _amp_context(device, amp):
    if amp and device.type != 'cpu':
        return autocast(device.type)
    return nullcontext()


def diff_dmc(volume, origin, spacing, direction, threshold):
    """
    volume: [X, Y, Z] torch.Tensor
    """
    import trimesh
    from diso import DiffDMC

    vertices, indices = DiffDMC(dtype=torch.float32)(-volume, None, isovalue=-threshold)
    vertices, indices = vertices.cpu().numpy(), indices.cpu().numpy()

    I = vertices * (np.array(volume.shape) - 1)

    direction = np.array(direction)[:3, :3]
    spacing = np.array(spacing)[:3]

    physical = (I * spacing) @ direction.T + origin
    return trimesh.Trimesh(physical, indices)


def _release_weight(filename, override=None):
    path = Path(override) if override else Path(__file__).parent / filename
    if not path.is_file():
        raise SystemError(f'Not found:\t {path.resolve()}')
    return path


def i1_load_models(
    vae_pre_path=None,
    vae_metal_path=None,
    rflow_path=None,
    metal_cls_path=None,
    cpu=False,
    printf=_printf,
):
    device = _device(cpu)
    printf('Device:\t {0}'.format(device))

    vae_dual = []
    for subtask, override in (('pre', vae_pre_path), ('metal', vae_metal_path)):
        weight = _release_weight(f'vae_{subtask}_best.pt', override)
        printf('VAE:\t [{0}] {1}'.format(subtask, weight.resolve()))

        loaded = torch.load(weight, map_location='cpu', weights_only=False)
        printf('Epoch:\t', loaded['epoch'])
        printf('Channels:\t', channels := loaded['channels'])
        printf('Scale Factor:\t', loaded['scale_factor'])
        printf('Global Mean:\t', loaded['global_mean'])

        vae_model = define.vae_kl(channels).to(device)
        vae_model.load_state_dict(loaded['state_dict'])
        vae_model.eval().float()
        printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in vae_model.parameters()) / 1e9))
        vae_dual.append((vae_model, loaded['scale_factor'], loaded['global_mean'], loaded['channels']))

    rflow_weight = _release_weight('rflow_last.pt', rflow_path)
    rflow_ckpt = torch.load(rflow_weight, map_location=device, weights_only=False)
    if 'condition_encoder_state_ema' not in rflow_ckpt and 'condition_encoder_state' not in rflow_ckpt:
        raise SystemError(f'{rflow_weight.resolve()} is not a StructuredProsthesisConditionEncoder checkpoint.')

    printf('RFlow:\t {0}'.format(rflow_weight.resolve()))
    printf('Epoch:\t {0}'.format(rflow_ckpt['epoch']))

    rflow_model = define.rflow_unet(context_embedding_size=256).to(device)
    condition_encoder = define.StructuredProsthesisConditionEncoder(embed_dim=256).to(device)

    rflow_state = rflow_ckpt.get('rflow_state_ema', rflow_ckpt['rflow_state'])
    encoder_state = rflow_ckpt.get('condition_encoder_state_ema', rflow_ckpt['condition_encoder_state'])
    rflow_model.load_state_dict(rflow_state)
    condition_encoder.load_state_dict(encoder_state)
    if 'rflow_state_ema' in rflow_ckpt:
        printf('Loaded RFlow EMA weights.')
    if 'condition_encoder_state_ema' in rflow_ckpt:
        printf('Loaded condition encoder EMA weights.')

    rflow_model.eval().float()
    condition_encoder.eval().float()
    printf('Param:\t {0:.2f} B'.format(sum(p.numel() for p in rflow_model.parameters()) / 1e9))

    cls_weight = _release_weight('metal_cls_generated_best.pt', metal_cls_path)

    printf('MetalCls:\t {0}'.format(cls_weight.resolve()))
    cls_ckpt = torch.load(cls_weight, map_location=device, weights_only=False)
    numeric_class_values = cls_ckpt.get('numeric_class_values') or build_numeric_bins()
    numeric_class_values = {name: tuple(numeric_class_values[name]) for name in NUMERIC_LABEL_NAMES}
    metal_cls = MetalGeometryCls({name: len(numeric_class_values[name]) for name in NUMERIC_LABEL_NAMES}).to(device)
    metal_cls.load_state_dict(cls_ckpt['model'])
    metal_cls.eval().float()
    printf('Epoch:\t {0}'.format(cls_ckpt.get('epoch')))
    printf('Domain:\t {0}'.format(cls_ckpt.get('domain')))
    printf('Best score:\t {0}'.format(cls_ckpt.get('best_score')))

    cls_meta = {
        'numeric_class_values': numeric_class_values,
        'epoch': cls_ckpt.get('epoch'),
        'domain': cls_ckpt.get('domain'),
        'best_score': cls_ckpt.get('best_score'),
        'path': cls_weight,
    }
    return *vae_dual, rflow_model, condition_encoder, metal_cls, cls_meta


def i2_encode_condition(
    stem_brand=None,
    stem_size=None,
    cup_outer=None,
    head_outer=None,
    head_offset=None,
    liner_offset=None,
    cpu=False,
):
    device = _device(cpu)
    encoded = define.PrepareProsthesisConditiond(keys=['context'])({
        'context': {
            'femoral_spec': [stem_brand or '', stem_size or ''],
            'cup_outer': cup_outer,
            'head_outer': head_outer,
            'head_offset': head_offset,
            'liner_offset': liner_offset,
        }
    })
    return (
        encoded['stem_model_id'].unsqueeze(0).to(device),
        encoded['stem_spec_id'].unsqueeze(0).to(device),
        encoded['numerics'].unsqueeze(0).to(device),
        encoded['masks'].unsqueeze(0).to(device),
    )


def _encode_nifti(path, vae_model, scale_factor, mean, channels, sw_batch_size=4, sw_overlap=0.25, cpu=False, amp=True):
    device = _device(cpu)
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'Image not found:\t {path.resolve()}')

    transforms = Compose([
        LoadImaged(keys=['image'], reader='ITKReader'),
        EnsureChannelFirstd(keys=['image'], channel_dim=-1 if channels > 1 else 'no_channel'),
        Lambdad(keys=['image'], func=lambda x: torch.clamp(x, min=-1.0, max=1.0)),
        SpatialPadd(keys=['image'], spatial_size=(128, 128, 128), constant_values=-1.0),
        DivisiblePadd(keys=['image'], k=16, constant_values=-1.0),
    ])
    data = cast(dict, transforms({'image': path.as_posix()}))
    tensor = data['image'].unsqueeze(0).to(device)

    def encode_predictor(z):
        with _amp_context(device, amp):
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

    return (encoded - mean) * scale_factor


def i3_pre_encode(pre_path, vae_model, scale_factor, mean, channels, sw_batch_size=4, sw_overlap=0.25, cpu=False, amp=True):
    pre_path = Path(pre_path)
    itk_img = itk.imread(pre_path.as_posix())
    pre_origin = list(itk.origin(itk_img))
    pre_spacing = list(itk.spacing(itk_img))
    pre_size = list(itk.size(itk_img))
    direction = itk.GetArrayFromMatrix(itk_img.GetDirection())
    encoded = _encode_nifti(pre_path, vae_model, scale_factor, mean, channels, sw_batch_size, sw_overlap, cpu, amp)
    return encoded, pre_origin, pre_spacing, pre_size, direction


def i3_metal_encode(cup_path, stem_path, vae_model, scale_factor, mean, channels, sw_batch_size=4, sw_overlap=0.25, cpu=False, amp=True):
    z_cup = _encode_nifti(cup_path, vae_model, scale_factor, mean, channels, sw_batch_size, sw_overlap, cpu, amp)
    z_stem = _encode_nifti(stem_path, vae_model, scale_factor, mean, channels, sw_batch_size, sw_overlap, cpu, amp)
    return torch.cat([z_cup, z_stem], dim=1)


def i4_rflow_sample(
    rflow_model,
    condition_encoder,
    pre_encoded,
    stem_model_id,
    stem_spec_id,
    numerics,
    masks,
    mode='bone_only',
    seed=None,
    ts=5,
    cpu=False,
    amp=True,
):
    device = _device(cpu)
    bone_latent, masks = define.apply_prosthesis_condition_mode(pre_encoded, masks, mode)
    with torch.no_grad():
        context = condition_encoder(stem_model_id, stem_spec_id, numerics, masks)

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    else:
        generator = None

    gen_shape = list(pre_encoded.shape)
    gen_shape[1] = 8
    generated = torch.randn(gen_shape, device=device, generator=generator)

    scheduler = define.scheduler_rflow()
    scheduler.set_timesteps(num_inference_steps=ts)
    all_timesteps = scheduler.timesteps.to(device)
    all_next_timesteps = torch.cat([all_timesteps[1:], torch.zeros(1, dtype=all_timesteps.dtype, device=device)])

    for t, next_t in zip(all_timesteps, all_next_timesteps):
        with torch.no_grad(), _amp_context(device, amp):
            model_input = torch.cat([generated, bone_latent], dim=1)
            timestep = t.expand(generated.shape[0]).to(device)
            velocity_pred = rflow_model(model_input, timestep, context=context)
        with torch.no_grad():
            generated, _ = scheduler.step(velocity_pred, t, generated, next_t)

    return generated


def _center_crop(tensor, roi_size):
    roi_size = tuple(int(x) for x in roi_size)
    slices = [slice(None)]
    for axis, size in enumerate(roi_size):
        src = int(tensor.shape[axis + 1])
        start = max(0, (src - size) // 2)
        stop = min(src, start + size)
        slices.append(slice(start, stop))
    return tensor[tuple(slices)]


def i5_metal_decode(generated, roi_size, vae_model, scale_factor, mean, channels, sw_batch_size=4, sw_overlap=0.25, cpu=False, amp=True):
    device = _device(cpu)
    z = generated / scale_factor + mean

    def decode_predictor(inputs: torch.Tensor) -> torch.Tensor:
        with _amp_context(device, amp):
            vae_latent_ch = vae_model.latent_channels
            if inputs.shape[1] > vae_latent_ch:
                recons = []
                for i in range(0, inputs.shape[1], vae_latent_ch):
                    recons.append(vae_model.decode(inputs[:, i : i + vae_latent_ch]))
                return torch.cat(recons, dim=1)
            return vae_model.decode(inputs)

    with torch.no_grad():
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Using a non-tuple sequence for multidimensional indexing')
            recon = sliding_window_inference(
                inputs=z,
                roi_size=(32, 32, 32),
                sw_batch_size=sw_batch_size,
                predictor=decode_predictor,
                overlap=sw_overlap,
                mode='gaussian',
                device=device,
                sw_device=device,
                progress=False,
            )

    decoded = recon[0].detach().cpu().float()
    decoded = _center_crop(decoded, roi_size)
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


@torch.no_grad()
def i7_classify_metal(metal_cls, metal_latent, numeric_class_values, cpu=False, amp=True):
    device = _device(cpu)
    if not torch.is_tensor(metal_latent):
        metal_latent = torch.from_numpy(np.asarray(metal_latent))
    if metal_latent.ndim == 4:
        metal_latent = metal_latent.unsqueeze(0)
    metal_latent = metal_latent.to(device=device, dtype=torch.float32)

    with _amp_context(device, amp):
        outputs = metal_cls(metal_latent)

    model_prob = torch.softmax(outputs['stem_model'].float(), dim=1)[0]
    topk = min(3, int(model_prob.numel()))
    top_p, top_i = model_prob.topk(topk)
    model_id = int(top_i[0])

    spec_logits = outputs['stem_spec'].float()
    allowed = build_stem_spec_model_mask(spec_logits.device)[model_id]
    if bool(allowed.any()):
        spec_logits = spec_logits.masked_fill(~allowed.unsqueeze(0), -1e9)
    spec_prob = torch.softmax(spec_logits, dim=1)[0]
    spec_id = int(spec_prob.argmax())
    spec_model, spec_size = define.FEMORAL_STEM_SPECS[spec_id]

    result = {
        'stem_model': {
            'label': define.FEMORAL_STEM_MODELS[model_id],
            'prob': float(model_prob[model_id]),
            'top3': [(define.FEMORAL_STEM_MODELS[int(index)], float(prob)) for prob, index in zip(top_p, top_i)],
        },
        'stem_spec': {
            'label': spec_size,
            'model': spec_model,
            'prob': float(spec_prob[spec_id]),
        },
    }
    for name in NUMERIC_LABEL_NAMES:
        dist = torch.softmax(outputs[name].float(), dim=1)[0]
        index = int(dist.argmax())
        values = list(numeric_class_values[name])
        result[name] = {'label': float(values[index]), 'prob': float(dist[index])}
    return result


if __name__ == '__main__':
    import tomlkit

    cfg = Path('config.toml')
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()

    prl = list(cfg['test'].keys())[1]
    pre_path = Path(cfg['train']['root']) / 'dataset' / 'pre' / f'{prl}.nii.gz'

    vae_pre, vae_metal, rflow, condition_encoder, metal_cls, cls_meta = i1_load_models()
    stem_model_id, stem_spec_id, numerics, masks = i2_encode_condition()
    pre_encoded, pre_origin, pre_spacing, pre_size, direction = i3_pre_encode(pre_path, *vae_pre)
    metal_latent = i4_rflow_sample(
        rflow,
        condition_encoder,
        pre_encoded,
        stem_model_id,
        stem_spec_id,
        numerics,
        masks,
        mode='bone_only',
    )
    pred = i7_classify_metal(metal_cls, metal_latent, cls_meta['numeric_class_values'])
    cup, stem = i5_metal_decode(metal_latent, pre_size, *vae_metal)
    i6_export('save_infer', cup, stem, pre_path, pre_origin, pre_spacing, direction)
    print(pre_size, cup.shape, stem.shape, pred['stem_model'])
