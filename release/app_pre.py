# uv run streamlit run release/app_pre.py --server.port 8506 --server.maxUploadSize 4096

import gc
import io
import math
import sys
import tempfile
import threading
import time
import warnings
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

warnings.filterwarnings('ignore', message='.*non-tuple sequence for multidimensional indexing.*')

_RELEASE = Path(__file__).resolve().parent
if str(_RELEASE) not in sys.path:
    sys.path.insert(0, str(_RELEASE))

from infer import (
    CONDITION_MODE_PARAM_LEVEL,
    NUMERIC_FALLBACKS,
    NUMERIC_LABEL_NAMES,
    NUMERIC_OPTIONS,
    STEM_MODELS,
    i1_load_models,
    i2_encode_condition,
    i3_pre_encode,
    i4_rflow_sample,
    i5_metal_decode,
    i6_export,
    i7_classify_metal,
    specs_for_model,
)
from preprocess import (
    SIDE_NAMES,
    available_sides,
    load_itk_xyz,
    prepare_preoperative_volume,
    run_totalsegmentator,
    side_voxel_counts,
)

st.set_page_config('THA-Flow', layout='wide', initial_sidebar_state='collapsed')

PROJECTIONS = {'正位': 1, '侧位': 0, '轴位': 2}
STAGES = ('影像', '解剖', '规划')
MODE_LABELS = {
    'bone_only': '骨骼形态',
    'bone_stem_model': '骨骼与柄型号',
    'bone_stem_model_spec': '骨骼与柄型号规格',
    'bone_prosthesis_full': '骨骼与假体参数',
    'prosthesis_full_only': '假体参数',
    'unconditional': '无条件',
}
MODE_ORDER = tuple(MODE_LABELS)


def _option_index(options, value, fallback=0):
    options = list(options)
    if value in options:
        return options.index(value)
    return fallback if 0 <= fallback < len(options) else 0


def _format_mm(name, value):
    if name in ('cup_outer', 'head_outer'):
        return f'{float(value):.0f} mm'
    if name == 'head_offset':
        return f'{float(value):+g} mm'
    return f'{float(value):g} mm'


def _format_condition_value(name, value):
    if value is None or value == '':
        return '—'
    if name in NUMERIC_LABEL_NAMES:
        return _format_mm(name, value)
    return str(value)


def _compare_rows(cond_values, pred):
    cond_values = cond_values or {}
    fields = [
        ('stem_model', '柄型号', 'stem_model'),
        ('stem_size', '柄规格', 'stem_spec'),
        ('cup_outer', '杯直径', 'cup_outer'),
        ('head_outer', '头直径', 'head_outer'),
        ('head_offset', '头偏距', 'head_offset'),
        ('liner_offset', '衬偏心', 'liner_offset'),
    ]
    rows = []
    for cond_key, label, pred_key in fields:
        cond_text = _format_condition_value(cond_key, cond_values.get(cond_key))
        if pred:
            item = pred[pred_key]
            pred_text = _format_mm(pred_key, item['label']) if pred_key in NUMERIC_LABEL_NAMES else (item['label'] or '—')
            prob_text = f'{item["prob"]:.1%}'
        else:
            pred_text = '—'
            prob_text = '—'
        rows.append((label, cond_text, pred_text, prob_text))
    return rows


def fast_drr(a, ax, th=(0.05, 1.0), mode: Literal['mean', 'max'] = 'mean'):
    a = np.asarray(a)
    c = th[0] < a
    a = a * c
    if mode == 'mean':
        a = a.sum(axis=ax)
        c = np.sum(c, axis=ax)
        c[c <= 0] = 1
        a = a / c
    else:
        a = a.max(axis=ax)
    return a


def _ct_display(image_xyz):
    return np.clip((image_xyz.astype(np.float32) - 150.0) / 500.0, 0.0, 1.0)


def _view_cache():
    return st.session_state.setdefault('_view_cache', {})


def _cache_put(key, value, limit=32):
    cache = _view_cache()
    cache[key] = value
    while len(cache) > limit:
        cache.pop(next(iter(cache)))
    return value


def render_slices(case_id, ax, white, green, blue):
    key = (case_id, 'slices', ax, id(white), id(green), id(blue))
    cache = _view_cache()
    if key in cache:
        return cache[key]
    axes = tuple(i for i in range(3) if i != ax)
    g_ax = np.any(green > 0.5, axis=axes)
    b_ax = np.any(blue > 0.5, axis=axes)
    g_indices = np.where(g_ax)[0]
    b_indices = np.where(b_ax)[0]
    g_min = g_indices[0] if len(g_indices) else 0
    g_max = g_indices[-1] if len(g_indices) else 0
    b_min = b_indices[0] if len(b_indices) else 0
    b_max = b_indices[-1] if len(b_indices) else 0
    if len(g_indices) and len(b_indices):
        k_min, k_max = min(g_min, b_min), max(g_max, b_max)
    else:
        k_min, k_max = g_min or b_min, g_max or b_max
    slices = []
    for k in range(k_max, k_min - 1, -1):
        w = np.take(white, k, axis=ax).transpose(1, 0)
        g = np.take(green, k, axis=ax).transpose(1, 0)
        b = np.take(blue, k, axis=ax).transpose(1, 0)
        if ax in (0, 1):
            w, g, b = np.flipud(w), np.flipud(g), np.flipud(b)
        rgb = np.stack([w, w, w], axis=-1)
        g_mask = g > 0.0
        rgb[g_mask] = (1.0 - rgb[g_mask]) * 0.5 + np.array([0.5, 1.0, 0.5]) * 0.5
        b_mask = b > 0.0
        rgb[b_mask] = (1.0 - rgb[b_mask]) * 0.5 + np.array([0.0, 0.5, 1.0]) * 0.5
        gb_mask = g_mask & b_mask
        rgb[gb_mask] = (1.0 - rgb[gb_mask]) * 0.5 + np.array([0.25, 0.75, 0.75]) * 0.5
        slices.append((k, (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)))
    return _cache_put(key, slices, limit=8)


@st.cache_resource(show_spinner=False)
def cached_models():
    return i1_load_models(printf=lambda *_: None)


def _workdir():
    if 'workdir' not in st.session_state:
        st.session_state['workdir'] = Path(tempfile.mkdtemp(prefix='nonahip_pre_'))
    return Path(st.session_state['workdir'])


def _reset(*keys):
    for key in keys:
        st.session_state.pop(key, None)


def _stage():
    if st.session_state.get('roi') is not None:
        return '规划'
    if st.session_state.get('seg') is not None:
        return '解剖'
    return '影像'


def _inject_chrome():
    st.html(
        """
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 2.4rem; max-width: 1440px;}
div[data-testid="stHeader"] {background: transparent;}
div[data-testid="stHeading"] h2 {margin: 0; padding: 0; font-size: 1.55rem; font-weight: 650; letter-spacing: .01em; line-height: 1.3;}
iframe[title="st.components.v1.html"] {position: absolute; width: 0; height: 0; border: 0; visibility: hidden;}
.tha-step {display:flex; gap:8px; align-items:center; margin: 0 0 1rem 0; line-height: 1.6;}
.tha-step span {padding:4px 11px; border-radius:999px; font-size:12px; color:#9ca3af; background:#f3f4f6;}
.tha-step span.on {color:#111827; background:#e5e7eb; font-weight:600;}
.tha-step span.done {color:#374151; background:#eef2ff;}
.tha-step i {color:#d1d5db; font-style:normal;}
</style>
        """
    )
    components.html(
        """
<script>
(function() {
  const w = window.parent || window;
  const d = w.document;
  if (w.__nonahipUploadHook) return;
  w.__nonahipUploadHook = true;

  function bar() {
    let el = d.getElementById('nonahip-upload-progress');
    if (!el) {
      el = d.createElement('div');
      el.id = 'nonahip-upload-progress';
      el.style.cssText = [
        'display:none',
        'position:fixed',
        'left:50%',
        'bottom:24px',
        'transform:translateX(-50%)',
        'z-index:999999',
        'background:#111827',
        'color:#fff',
        'padding:12px 16px',
        'border-radius:10px',
        'box-shadow:0 10px 30px rgba(0,0,0,.28)',
        'font:13px/1.4 system-ui,sans-serif',
        'min-width:320px'
      ].join(';');
      d.body.appendChild(el);
    }
    return el;
  }

  function show(loaded, total) {
    if (!total) return;
    const el = bar();
    const pct = Math.max(0, Math.min(100, loaded / total * 100));
    el.style.display = 'block';
    el.innerHTML = '<div>上传影像</div><div style="margin-top:8px;height:8px;background:#374151;border-radius:99px;overflow:hidden"><div style="height:100%;width:' + pct.toFixed(1) + '%;background:#60a5fa"></div></div><div style="margin-top:8px;opacity:.9">' + (loaded / 1048576).toFixed(0) + ' / ' + (total / 1048576).toFixed(0) + ' MB（' + pct.toFixed(0) + '%）</div>';
    if (loaded >= total) setTimeout(function() { el.style.display = 'none'; }, 1200);
  }

  const send = w.XMLHttpRequest.prototype.send;
  w.XMLHttpRequest.prototype.send = function() {
    this.upload.addEventListener('progress', function(ev) {
      if (ev.lengthComputable) show(ev.loaded, ev.total);
    });
    return send.apply(this, arguments);
  };
})();
</script>
        """,
        height=0,
        width=0,
    )


def _save_uploaded_file(uploaded, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(getattr(uploaded, 'size', 0) or 0)
    if hasattr(uploaded, 'seek'):
        uploaded.seek(0)
    bar = st.progress(0.0, '保存影像')
    written = 0
    with dest.open('wb') as handle:
        while True:
            chunk = uploaded.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if total > 0:
                bar.progress(min(1.0, written / total), f'保存影像  {written / 1048576:.0f} / {total / 1048576:.0f} MB')
            else:
                bar.progress(0.0, f'保存影像  {written / 1048576:.0f} MB')
    bar.empty()
    if hasattr(uploaded, 'seek'):
        uploaded.seek(0)


def run_with_progress(label, fn, hint_s=45, box=None):
    box = box if box is not None else st.empty()
    holder = {}

    def worker():
        try:
            holder['value'] = fn()
        except Exception as exc:
            holder['error'] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    t0 = time.time()
    while thread.is_alive():
        elapsed = time.time() - t0
        frac = min(0.97, 1.0 - math.exp(-elapsed / max(hint_s, 1.0)))
        box.progress(frac, f'{label}  {elapsed:.0f} s')
        thread.join(0.25)
    thread.join()
    box.empty()
    if 'error' in holder:
        raise holder['error']
    return holder.get('value')


def render_drr(case_id, ax, white, green=None, blue=None):
    key = (case_id, 'drr', ax, id(white), id(green) if green is not None else 0, id(blue) if blue is not None else 0)
    cache = _view_cache()
    if key in cache:
        return cache[key]
    w = fast_drr(white, ax).transpose(1, 0)
    if ax in (0, 1):
        w = np.flipud(w)
    rgb = np.stack([w, w, w], axis=-1)
    if green is not None:
        g = fast_drr(green, ax).transpose(1, 0)
        if ax in (0, 1):
            g = np.flipud(g)
        g_mask = g > 0.0
        rgb[g_mask] = (1.0 - rgb[g_mask]) * 0.5 + np.array([0.5, 1.0, 0.5]) * 0.5
    else:
        g_mask = None
    if blue is not None:
        b = fast_drr(blue, ax).transpose(1, 0)
        if ax in (0, 1):
            b = np.flipud(b)
        b_mask = b > 0.0
        rgb[b_mask] = (1.0 - rgb[b_mask]) * 0.5 + np.array([0.0, 0.5, 1.0]) * 0.5
        if g_mask is not None:
            gb_mask = g_mask & b_mask
            rgb[gb_mask] = (1.0 - rgb[gb_mask]) * 0.5 + np.array([0.25, 0.75, 0.75]) * 0.5
    return _cache_put(key, (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))


def _show_drr(case_id, white, green=None, blue=None, legend=None, key=None):
    proj = st.segmented_control(
        '投影',
        list(PROJECTIONS),
        default='正位',
        required=True,
        key=key or f'{case_id}_proj',
        label_visibility='collapsed',
        width='stretch',
    )
    ax = PROJECTIONS[proj]
    st.image(render_drr(case_id, ax, white, green, blue), width='stretch')
    if legend:
        st.caption(legend)
    return ax


@st.fragment
def image_viewer():
    ct = st.session_state['ct']
    case_id = st.session_state.get('case_name') or 'case'
    _show_drr(f'{case_id}-raw', ct['white'], key=f'{case_id}_raw_proj')


@st.fragment
def anatomy_viewer():
    ct = st.session_state['ct']
    seg = st.session_state['seg']
    case_id = st.session_state.get('case_name') or 'case'
    _show_drr(
        f'{case_id}-seg',
        ct['white'],
        seg['overlay_l'],
        seg['overlay_r'],
        legend='绿：左侧　蓝：右侧',
        key=f'{case_id}_seg_proj',
    )


@st.fragment
def anatomy_panel():
    seg = st.session_state['seg']
    for side in ('L', 'R'):
        st.metric(SIDE_NAMES[side], '可规划' if seg['counts'][side]['ok'] else '未识别')
    side = st.radio('术侧', seg['sides'], format_func=lambda key: SIDE_NAMES[key], horizontal=True, key='plan_side')
    if st.button('提取术区', type='primary', width='stretch'):
        st.session_state['_action'] = ('extract', side)
        st.rerun()
    if st.button('重新分割', width='stretch'):
        st.session_state['_action'] = 'resegment'
        st.rerun()


@st.fragment
def plan_view():
    roi = st.session_state['roi']
    generated = st.session_state.get('generated') or []
    case_id = st.session_state.get('case_name') or 'case'
    pre_white = roi['white']
    item = None
    cup = stem = None
    if generated:
        labels = [str(i + 1) for i in range(len(generated))]
        key = f'{case_id}_plan_id'
        if st.session_state.get(key) not in labels:
            st.session_state[key] = labels[-1]
        choice = st.segmented_control('方案', labels, required=True, key=key)
        metal_id = labels.index(choice)
        item = generated[metal_id]
        cup, stem = item['cup'], item['stem']
    ax = _show_drr(
        f'{case_id}-plan',
        pre_white,
        cup,
        stem,
        legend='绿：髋臼侧　蓝：股骨侧' if item is not None else None,
        key=f'{case_id}_plan_proj',
    )
    if item is None:
        return
    rows = _compare_rows(item.get('condition') or {}, item.get('pred'))
    st.dataframe(
        {
            '参数': [row[0] for row in rows],
            '设定': [row[1] for row in rows],
            '识别': [row[2] for row in rows],
            '置信度': [row[3] for row in rows],
        },
        hide_index=True,
        width='stretch',
    )
    if st.toggle('断层', value=False, key=f'{case_id}_slices_on'):
        cn = st.number_input('列', 1, 12, 8, 1, key=f'{case_id}_slice_cols')
        slices = render_slices(f'{case_id}-plan', ax, pre_white, cup, stem)
        for i in range(0, len(slices), cn):
            cols = st.columns(cn)
            for j, col in enumerate(cols):
                if i + j < len(slices):
                    col.image(slices[i + j][1], caption=str(slices[i + j][0]))


@st.fragment
def plan_controls():
    mode = st.selectbox('规划依据', MODE_ORDER, format_func=lambda key: MODE_LABELS[key], key='plan_mode')
    param_level = CONDITION_MODE_PARAM_LEVEL[mode]
    stem_brand = stem_size = cup_outer = head_outer = head_offset = liner_offset = None
    if param_level in ('model', 'model_spec', 'full'):
        stem_brand = st.selectbox('柄型号', STEM_MODELS, key='plan_stem_brand')
    if param_level in ('model_spec', 'full') and stem_brand is not None:
        spec_options = specs_for_model(stem_brand)
        if spec_options:
            stem_size = st.selectbox('柄规格', spec_options, key='plan_stem_size')
    if param_level == 'full':
        cup_outer = st.selectbox(
            '杯直径',
            NUMERIC_OPTIONS['cup_outer'],
            index=_option_index(NUMERIC_OPTIONS['cup_outer'], NUMERIC_FALLBACKS['cup_outer']),
            format_func=lambda value: _format_mm('cup_outer', value),
            key='plan_cup_outer',
        )
        head_outer = st.selectbox(
            '头直径',
            NUMERIC_OPTIONS['head_outer'],
            index=_option_index(NUMERIC_OPTIONS['head_outer'], NUMERIC_FALLBACKS['head_outer']),
            format_func=lambda value: _format_mm('head_outer', value),
            key='plan_head_outer',
        )
        head_offset = st.selectbox(
            '头偏距',
            NUMERIC_OPTIONS['head_offset'],
            index=_option_index(NUMERIC_OPTIONS['head_offset'], NUMERIC_FALLBACKS['head_offset']),
            format_func=lambda value: _format_mm('head_offset', value),
            key='plan_head_offset',
        )
        liner_offset = st.selectbox(
            '衬偏心',
            NUMERIC_OPTIONS['liner_offset'],
            index=_option_index(NUMERIC_OPTIONS['liner_offset'], NUMERIC_FALLBACKS['liner_offset']),
            format_func=lambda value: _format_mm('liner_offset', value),
            key='plan_liner_offset',
        )
    with st.expander('高级'):
        seed = st.number_input('种子', 0, None, 42, 1, key='plan_seed')
    seed = int(st.session_state.get('plan_seed', 42))
    gen_l, gen_r = st.columns(2)
    if gen_l.button('生成', type='primary', width='stretch'):
        st.session_state['_action'] = (
            'generate',
            {
                'mode': mode,
                'stem_model': stem_brand,
                'stem_size': stem_size,
                'cup_outer': cup_outer,
                'head_outer': head_outer,
                'head_offset': head_offset,
                'liner_offset': liner_offset,
                'seed': seed,
            },
        )
        st.rerun()
    if gen_r.button('清空', width='stretch'):
        st.session_state['_action'] = 'clear_generated'
        st.rerun()
    if st.button('重选术侧', width='stretch'):
        st.session_state['_action'] = 'reselect_side'
        st.rerun()
    generated = st.session_state.get('generated') or []
    if generated:
        selected = st.multiselect('导出', ['术前图像', '术前骨骼模型', '假体距离场', '假体模型'], default=['假体模型'], key='export_items')
        if st.button('导出', width='stretch') and selected:
            st.session_state['_action'] = ('export', list(selected))
            st.rerun()
        bundle = st.session_state.get('export_bundle')
        if bundle:
            st.download_button('下载', data=bundle['data'], file_name=bundle['name'], mime='application/zip', width='stretch')


def _ingest_upload(uploaded):
    name = uploaded.name
    if name.endswith('.nii.gz') or name.endswith('.nii'):
        pass
    elif name.endswith('.gz'):
        name = name.removesuffix('.gz') + '.nii.gz'
    else:
        st.error(f'无法读取 {name}')
        return
    stamp = f'{name}:{uploaded.size}'
    if st.session_state.get('upload_stamp') == stamp and st.session_state.get('ct') is not None:
        return
    _reset('ct', 'seg', 'roi', 'generated', 'export_bundle', 'upload_stamp', 'case_name', '_view_cache')
    suffix = '.nii.gz' if name.endswith('.nii.gz') else '.nii'
    image_path = _workdir() / f'pre_raw{suffix}'
    _save_uploaded_file(uploaded, image_path)
    bar = st.progress(0.2, '读取影像')
    image_xyz, origin, spacing = load_itk_xyz(image_path)
    bar.progress(0.8, '准备预览')
    st.session_state['ct'] = {
        'path': image_path,
        'image': image_xyz,
        'origin': origin,
        'spacing': spacing,
        'white': _ct_display(image_xyz),
    }
    st.session_state['upload_stamp'] = stamp
    st.session_state['case_name'] = Path(name).name.removesuffix('.nii.gz').removesuffix('.nii')
    st.session_state['replace_ct'] = False
    bar.empty()
    st.rerun()


def _do_segment(work):
    ct = st.session_state['ct']
    image_path = Path(ct['path'])
    label_path = _workdir() / 'total.nii.gz'
    run_with_progress('解剖分割', lambda: run_totalsegmentator(image_path, label_path), hint_s=90, box=work)
    bar = work.progress(0.62, '读取分割')
    label_xyz, _, _ = load_itk_xyz(label_path)
    if label_xyz.shape != ct['image'].shape:
        raise RuntimeError('分割与影像尺寸不一致')
    bar.progress(0.88, '统计术侧')
    st.session_state['seg'] = {
        'label_path': label_path,
        'label': label_xyz,
        'counts': side_voxel_counts(label_xyz),
        'sides': available_sides(label_xyz),
        'overlay_l': ((label_xyz == 75) | (label_xyz == 77)).astype(np.float32),
        'overlay_r': ((label_xyz == 76) | (label_xyz == 78)).astype(np.float32),
    }
    bar.empty()


def _do_extract(work, side):
    ct = st.session_state['ct']
    seg = st.session_state['seg']
    case_name = st.session_state.get('case_name', 'case')
    pre_path = _workdir() / f'{case_name}_{side}_pre.nii.gz'
    t0 = time.time()
    bar = work.progress(0.0, '提取术区')
    prepared = prepare_preoperative_volume(
        ct['image'],
        seg['label'],
        ct['origin'],
        ct['spacing'],
        side,
        pre_path,
        progress=lambda value, text: bar.progress(float(value), f'{text}  {time.time() - t0:.0f} s'),
    )
    bar.empty()
    st.session_state['roi'] = {
        'side': side,
        'path': prepared['path'],
        'origin': prepared['origin'],
        'spacing': prepared['spacing'],
        'size': prepared['size'],
        'white': np.clip(np.asarray(prepared['image'], dtype=np.float32) * 0.5 + 0.5, 0.0, 1.0),
    }
    _reset('generated', 'export_bundle')


def _do_generate(work, condition):
    roi = st.session_state['roi']
    pre_path = Path(roi['path'])
    pre_size = list(roi['size'])
    t0 = time.time()
    bar = work.progress(0.0, '载入模型')

    def tick(frac, text):
        bar.progress(frac, f'{text}  {time.time() - t0:.0f} s')

    tick(0.08, '载入模型')
    vae_pre, vae_metal, rflow, condition_encoder, metal_cls, cls_meta = cached_models()
    tick(0.22, '编码条件')
    stem_model_id, stem_spec_id, numerics, masks = i2_encode_condition(
        condition['stem_model'],
        condition['stem_size'],
        condition['cup_outer'],
        condition['head_outer'],
        condition['head_offset'],
        condition['liner_offset'],
    )
    tick(0.36, '编码骨骼')
    pre_encoded, *_ = i3_pre_encode(pre_path, *vae_pre)
    tick(0.52, '生成假体')
    metal_latent = i4_rflow_sample(
        rflow,
        condition_encoder,
        pre_encoded,
        stem_model_id,
        stem_spec_id,
        numerics,
        masks,
        mode=condition['mode'],
        seed=condition['seed'],
        ts=5,
    )
    tick(0.74, '识别参数')
    pred = i7_classify_metal(metal_cls, metal_latent, cls_meta['numeric_class_values'])
    tick(0.84, '重建几何')
    cup_tsdf, stem_tsdf = i5_metal_decode(metal_latent, pre_size, *vae_metal)
    del metal_latent, pre_encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bar.empty()
    generated = list(st.session_state.get('generated', []))
    generated.append({'cup': cup_tsdf, 'stem': stem_tsdf, 'pred': pred, 'condition': condition})
    st.session_state['generated'] = generated
    case_name = st.session_state.get('case_name') or 'case'
    st.session_state[f'{case_name}_plan_id'] = str(len(generated))
    _reset('export_bundle')


def _do_export(work, selected):
    roi = st.session_state['roi']
    generated = st.session_state.get('generated') or []
    if not generated:
        return
    case_name = st.session_state.get('case_name') or 'case'
    metal_id = st.session_state.get(f'{case_name}_plan_id', str(len(generated)))
    try:
        metal_id = int(metal_id) - 1
    except (TypeError, ValueError):
        metal_id = len(generated) - 1
    if metal_id not in range(len(generated)):
        metal_id = len(generated) - 1
    item = generated[metal_id]
    pre_path = Path(roi['path'])
    pre_origin = list(roi['origin'])
    pre_spacing = list(roi['spacing'])
    pre_direction = np.eye(3, dtype=np.float64)
    t0 = time.time()
    bar = work.progress(0.25, '导出')
    with tempfile.TemporaryDirectory() as tempdir:
        savedir = Path(tempdir) / f'{case_name}_{roi["side"]}_{metal_id + 1}'
        i6_export(savedir, item['cup'], item['stem'], pre_path, pre_origin, pre_spacing, pre_direction)
        bar.progress(0.8, f'打包  {time.time() - t0:.0f} s')
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            if '术前图像' in selected:
                zf.write(pre_path, arcname='pre.nii.gz')
            if '术前骨骼模型' in selected:
                zf.write(savedir / 'pre.stl', arcname='pre.stl')
            if '假体距离场' in selected:
                zf.write(savedir / 'cup.nii.gz', arcname='cup.nii.gz')
                zf.write(savedir / 'stem.nii.gz', arcname='stem.nii.gz')
            if '假体模型' in selected:
                zf.write(savedir / 'cup.stl', arcname='cup.stl')
                zf.write(savedir / 'stem.stl', arcname='stem.stl')
                zf.write(savedir / 'metal.nii.gz', arcname='metal.nii.gz')
        st.session_state['export_bundle'] = {'data': memory_file.getvalue(), 'name': f'{savedir.name}.zip'}
    bar.empty()


_inject_chrome()

ct = st.session_state.get('ct')
stage = _stage()
case_name = st.session_state.get('case_name')
roi = st.session_state.get('roi')

head_l, head_r = st.columns([3, 2], vertical_alignment='bottom')
with head_l:
    st.markdown('## THA-Flow')
with head_r:
    bits = []
    if case_name:
        bits.append(case_name)
    if roi:
        bits.append(SIDE_NAMES.get(roi['side'], roi['side']))
    if bits:
        st.caption('  ·  '.join(bits))

step_bits = []
reached = STAGES.index(stage)
for i, name in enumerate(STAGES):
    cls = 'on' if i == reached else ('done' if i < reached else '')
    step_bits.append(f'<span class="{cls}">{name}</span>')
st.html('<div class="tha-step">' + '<i>—</i>'.join(step_bits) + '</div>')

work = st.empty()
action = st.session_state.pop('_action', None)
if action == 'segment':
    _do_segment(work)
    st.rerun()
elif action == 'replace':
    _reset('upload_stamp', 'ct', 'seg', 'roi', 'generated', 'export_bundle', 'case_name', '_view_cache')
    st.session_state['replace_ct'] = True
    st.rerun()
elif action == 'resegment':
    _reset('seg', 'roi', 'generated', 'export_bundle')
    st.rerun()
elif isinstance(action, tuple) and action[0] == 'extract':
    _do_extract(work, action[1])
    st.rerun()
elif isinstance(action, tuple) and action[0] == 'generate':
    _do_generate(work, action[1])
    st.rerun()
elif action == 'clear_generated':
    _reset('generated', 'export_bundle')
    plan_key = f'{st.session_state.get("case_name") or "case"}_plan_id'
    st.session_state.pop(plan_key, None)
    st.rerun()
elif action == 'reselect_side':
    _reset('roi', 'generated', 'export_bundle')
    st.rerun()
elif isinstance(action, tuple) and action[0] == 'export':
    _do_export(work, action[1])
    st.rerun()

show_uploader = ct is None or st.session_state.get('replace_ct')
if show_uploader:
    uploaded = st.file_uploader('术前 CT', type=['nii', 'gz'], accept_multiple_files=False)
    if uploaded is not None:
        _ingest_upload(uploaded)
    if ct is None:
        st.stop()

if ct is None:
    st.stop()

seg = st.session_state.get('seg')
if seg is None:
    view, panel = st.columns([7.2, 2.8], gap='large')
    with view:
        image_viewer()
    with panel:
        if st.button('解剖分割', type='primary', width='stretch'):
            st.session_state['_action'] = 'segment'
            st.rerun()
        if st.button('更换影像', width='stretch'):
            st.session_state['_action'] = 'replace'
            st.rerun()
    st.stop()

if not seg['sides']:
    st.error('未同时识别到髋骨与股骨')
    if st.button('重新分割'):
        _reset('seg', 'roi', 'generated', 'export_bundle')
        st.rerun()
    st.stop()

if roi is None:
    view, panel = st.columns([7.2, 2.8], gap='large')
    with view:
        anatomy_viewer()
    with panel:
        anatomy_panel()
    st.stop()

view, panel = st.columns([7.2, 2.8], gap='large')
with view:
    plan_view()
with panel:
    plan_controls()
