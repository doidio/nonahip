# uv run streamlit run release/app.py --server.port 8505 -- --config config.toml

import argparse
import gc
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import itk
import numpy as np
import streamlit as st
import tomlkit
import torch
from infer import (
    CONDITION_MODE_LABELS,
    CONDITION_MODE_PARAM_LEVEL,
    NUMERIC_FALLBACKS,
    NUMERIC_LABEL_NAMES,
    NUMERIC_OPTIONS,
    STEM_MODELS,
    case_prosthesis_values,
    format_condition_desc,
    i1_load_models,
    i2_encode_condition,
    i3_metal_encode,
    i3_pre_encode,
    i4_rflow_sample,
    i5_metal_decode,
    i6_export,
    i7_classify_metal,
    nearest_numeric,
    specs_for_model,
)

st.set_page_config('Nonahip', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonahip 假体预测生成')


# @st.cache_resource(show_spinner=False)
def cache_load_pairs(config_file: str):
    cfg = Path(config_file)
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()

    root = Path(cfg['dataset']['root'])
    excluded = cfg['pairs']['excluded']

    categories = ['pair', 'roi', 'align', 'context']

    tests = {}
    for prl in cfg['test']:
        if prl in excluded:
            continue

        pid, rl = prl.split('_')
        parent = root / 'pair' / pid / rl

        if not (parent / 'pair.toml').exists():
            continue

        it = {}

        for category in set(['pair'] + categories):
            f = parent / f'{category}.toml'
            if f.exists():
                data = tomlkit.loads(f.read_text('utf-8')).unwrap()
                it[category] = data
            else:
                it[category] = {}

        it['prl'] = prl
        tests[prl] = it
    return cfg, tests


def it_desc(it):
    values = case_prosthesis_values(it.get('context', {}))
    return format_condition_desc(seed=it.get('seed'), **values)


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
            pred_text = _format_mm(pred_key, item['label']) if pred_key in NUMERIC_LABEL_NAMES else (item['label'] or '未标注')
            prob_text = f'{item["prob"]:.1%}'
        else:
            pred_text = '—'
            prob_text = '—'
        rows.append((label, cond_text, pred_text, prob_text))
    return rows


def fast_drr(a, ax, th=(0.05, 1.0), mode: Literal['mean', 'max'] = 'mean'):
    a = a.copy()
    c = th[0] < a
    a *= c
    if mode == 'mean':
        a = a.sum(axis=ax)
        c = np.sum(c, axis=ax)
        c[np.where(c <= 0)] = 1
        a = a / c
    elif mode == 'max':
        a = a.max(axis=ax)

    return a


@st.cache_data(show_spinner='正在作图', show_time=True)
def render_drr(prl, canvas_t, ax, white, green, blue) -> list:
    w = fast_drr(white, ax).transpose(1, 0)
    g = fast_drr(green, ax).transpose(1, 0)
    b = fast_drr(blue, ax).transpose(1, 0)

    if ax in (0, 1):
        w, g, b = np.flipud(w), np.flipud(g), np.flipud(b)

    rgb = np.stack([w, w, w], axis=-1)

    g_mask = g > 0.0
    rgb[g_mask] = (1.0 - rgb[g_mask]) * 0.5 + np.array([0.5, 1.0, 0.5]) * 0.5

    b_mask = b > 0.0
    rgb[b_mask] = (1.0 - rgb[b_mask]) * 0.5 + np.array([0.0, 0.5, 1.0]) * 0.5

    gb_mask = g_mask & b_mask
    rgb[gb_mask] = (1.0 - rgb[gb_mask]) * 0.5 + np.array([0.25, 0.75, 0.75]) * 0.5

    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return [(0, rgb_uint8)]


@st.cache_data(show_spinner='正在作图', show_time=True)
def render_slices(prl, canvas_t, ax, white, green, blue) -> list:
    axes = tuple(i for i in range(3) if i != ax)
    g_ax = np.any(green > 0.5, axis=axes)
    b_ax = np.any(blue > 0.5, axis=axes)

    g_indices = np.where(g_ax)[0]
    b_indices = np.where(b_ax)[0]

    g_min = g_indices[0] if len(g_indices) > 0 else 0
    g_max = g_indices[-1] if len(g_indices) > 0 else 0
    b_min = b_indices[0] if len(b_indices) > 0 else 0
    b_max = b_indices[-1] if len(b_indices) > 0 else 0

    if len(g_indices) > 0 and len(b_indices) > 0:
        k_min, k_max = min(g_min, b_min), max(g_max, b_max)
    else:
        k_min, k_max = g_min or b_min, g_max or b_max

    kn = k_max - k_min + 1
    slices = []
    for i in range(kn):
        k = k_max - i
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

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        slices.append((k, rgb_uint8))
    return slices


if (it := st.session_state.get('init')) is None:
    render_slices.clear()

    with st.spinner('初始化', show_time=True):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args, _ = parser.parse_known_args()
        cfg, tests = cache_load_pairs(args.config)

    st.session_state['init'] = cfg, tests
    st.rerun()

elif (it := st.session_state.get('prl')) is None:
    cfg, tests = st.session_state['init']

    if len(tests) == 0:
        st.warning('测试集为空')
        st.stop()

    prl = st.selectbox('测试集', list(sorted(tests.keys())), format_func=lambda _: f'{_} {it_desc(tests[_])}')

    dataset = Path(cfg['train']['root']) / 'dataset'

    if st.button('载入', width=200):
        with st.spinner('正在载入', show_time=True):
            pre = dataset / 'pre' / f'{prl}.nii.gz'
            post_align_hip = dataset / 'post_align_hip' / f'{prl}.nii.gz'
            post_align_femur = dataset / 'post_align_femur' / f'{prl}.nii.gz'
            cup = dataset / 'metal' / f'{prl}_cup.nii.gz'
            stem = dataset / 'metal' / f'{prl}_stem.nii.gz'

            images = []
            for i, f in enumerate((pre, post_align_hip, post_align_femur, cup, stem)):
                img = itk.imread(f.as_posix())

                if i == 0:
                    pre_origin = list(itk.origin(img))
                    pre_spacing = list(itk.spacing(img))
                    pre_size = list(itk.size(img))
                    pre_direction = itk.GetArrayFromMatrix(img.GetDirection())
                    st.session_state['pre'] = pre_origin, pre_spacing, pre_size, pre_direction

                img = itk.array_from_image(img).transpose(2, 1, 0)
                if i >= 3:
                    img = np.where(img > 0.0, 1.0, 0.0)
                else:
                    img = img * 0.5 + 0.5
                images.append(img)

        st.session_state['prl'] = prl, *images

        st.rerun()

    st.code(tomlkit.dumps(tests[prl]), 'toml')

else:
    cfg, tests = st.session_state['init']
    prl, pre, post_align_hip, post_align_femur, cup, stem = st.session_state['prl']
    pre_origin, pre_spacing, pre_size, pre_direction = st.session_state['pre']
    it = tests[prl]
    desc = it_desc(it)

    pre_path = Path(cfg['train']['root']) / 'dataset' / 'pre' / f'{prl}.nii.gz'

    with st.expander(f'{prl} {desc}', expanded=False):
        st.code(tomlkit.dumps(it), 'toml')

    cols = st.columns([1, 3])

    top_cols = cols[1].columns([2, 1, 1], vertical_alignment='bottom')
    sub_cols = cols[1].columns([4, 1, 1])

    stx = st.container()
    log = st.expander('日志', expanded=True)

    canvas = {'术前': pre, '术后对齐骨盆': post_align_hip, '术后对齐股骨': post_align_femur}
    canvas_t = cols[0].radio('空间', list(canvas.keys()), horizontal=False)
    canvas = canvas[canvas_t]

    ax = {'正位': 1, '侧位': 0, '轴位': 2}
    ax_t = cols[0].radio('方位', list(ax.keys()), horizontal=False)
    ax = ax[ax_t]

    defaults = case_prosthesis_values(it.get('context', {}))
    mode = sub_cols[0].selectbox(
        '条件模式',
        list(CONDITION_MODE_LABELS.keys()),
        index=list(CONDITION_MODE_LABELS).index('bone_only'),
        format_func=lambda key: CONDITION_MODE_LABELS[key],
    )
    param_level = CONDITION_MODE_PARAM_LEVEL[mode]
    sub_cols[0].caption('与训练时的六种条件接口一致。')

    stem_brand = stem_size = cup_outer = head_outer = head_offset = liner_offset = None

    if param_level in ('model', 'model_spec', 'full'):
        stem_brand = sub_cols[0].selectbox(
            '柄型号',
            STEM_MODELS,
            index=_option_index(STEM_MODELS, defaults['stem_model']),
        )
    if param_level in ('model_spec', 'full') and stem_brand is not None:
        spec_options = specs_for_model(stem_brand)
        if spec_options:
            stem_size = sub_cols[0].selectbox(
                '柄规格',
                spec_options,
                index=_option_index(spec_options, defaults['stem_size']),
            )

    if param_level == 'full':
        cup_outer = sub_cols[1].selectbox(
            '杯直径',
            NUMERIC_OPTIONS['cup_outer'],
            index=_option_index(
                NUMERIC_OPTIONS['cup_outer'],
                nearest_numeric(defaults['cup_outer'], NUMERIC_OPTIONS['cup_outer'], NUMERIC_FALLBACKS['cup_outer']),
            ),
            format_func=lambda value: _format_mm('cup_outer', value),
        )
        head_outer = sub_cols[1].selectbox(
            '头直径',
            NUMERIC_OPTIONS['head_outer'],
            index=_option_index(
                NUMERIC_OPTIONS['head_outer'],
                nearest_numeric(defaults['head_outer'], NUMERIC_OPTIONS['head_outer'], NUMERIC_FALLBACKS['head_outer']),
            ),
            format_func=lambda value: _format_mm('head_outer', value),
        )
        head_offset = sub_cols[2].selectbox(
            '头偏距',
            NUMERIC_OPTIONS['head_offset'],
            index=_option_index(
                NUMERIC_OPTIONS['head_offset'],
                nearest_numeric(defaults['head_offset'], NUMERIC_OPTIONS['head_offset'], NUMERIC_FALLBACKS['head_offset']),
            ),
            format_func=lambda value: _format_mm('head_offset', value),
        )
        liner_offset = sub_cols[2].selectbox(
            '衬偏心',
            NUMERIC_OPTIONS['liner_offset'],
            index=_option_index(
                NUMERIC_OPTIONS['liner_offset'],
                nearest_numeric(defaults['liner_offset'], NUMERIC_OPTIONS['liner_offset'], NUMERIC_FALLBACKS['liner_offset']),
            ),
            format_func=lambda value: _format_mm('liner_offset', value),
        )
    elif param_level is None:
        sub_cols[1].caption('此模式不注入假体参数，由分类器从生成几何中读出。')

    seed = top_cols[2].number_input('随机种子', 0, None, 42, 1)

    if top_cols[1].button('清空'):
        st.session_state.pop('generated', None)
        st.session_state.pop('real_pred', None)
        st.rerun()

    if top_cols[0].button(f'{CONDITION_MODE_LABELS[mode]}生成', width='stretch'):
        condition = {
            'mode': mode,
            'stem_model': stem_brand,
            'stem_size': stem_size,
            'cup_outer': cup_outer,
            'head_outer': head_outer,
            'head_offset': head_offset,
            'liner_offset': liner_offset,
            'seed': seed,
        }

        bar = stx.progress(0.0, '载入模型')
        vae_pre, vae_metal, rflow, condition_encoder, metal_cls, cls_meta = i1_load_models(
            printf=lambda *args: log.caption('\t'.join(str(_) for _ in args)),
        )

        bar.progress(0.15, '编码假体条件')
        stem_model_id, stem_spec_id, numerics, masks = i2_encode_condition(stem_brand, stem_size, cup_outer, head_outer, head_offset, liner_offset)

        bar.progress(0.25, '编码术前图像')
        pre_encoded, *_ = i3_pre_encode(pre_path, *vae_pre)

        metal_dir = Path(cfg['train']['root']) / 'dataset' / 'metal'
        cup_path = metal_dir / f'{prl}_cup.nii.gz'
        stem_path = metal_dir / f'{prl}_stem.nii.gz'
        if cup_path.exists() and stem_path.exists():
            bar.progress(0.32, '判别真实假体')
            real_latent = i3_metal_encode(cup_path, stem_path, *vae_metal)
            st.session_state['real_pred'] = i7_classify_metal(metal_cls, real_latent, cls_meta['numeric_class_values'])
            del real_latent

        bar.progress(0.45, '采样假体')
        metal_latent = i4_rflow_sample(
            rflow,
            condition_encoder,
            pre_encoded,
            stem_model_id,
            stem_spec_id,
            numerics,
            masks,
            mode=mode,
            seed=seed,
            ts=5,
        )
        bar.progress(0.7, '判别假体')
        pred = i7_classify_metal(metal_cls, metal_latent, cls_meta['numeric_class_values'])
        bar.progress(0.85, '解码假体')
        cup_tsdf, stem_tsdf = i5_metal_decode(metal_latent, pre_size, *vae_metal)
        del metal_latent

        bar.empty()

        del vae_pre, vae_metal, rflow, condition_encoder, metal_cls, pre_encoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        st.session_state['generated'].append(
            {
                'cup': cup_tsdf,
                'stem': stem_tsdf,
                'pred': pred,
                'condition': condition,
            }
        )

    def format_func(i):
        if i > 0:
            return f'预测 {i}'
        return '真实'

    generated = st.session_state.get('generated', [])
    metal_id = stx.radio('假体', range(len(generated) + 1), horizontal=True, format_func=format_func)

    pred = st.session_state.get('real_pred')
    cond_values = defaults
    cond_title = '手术记录'
    if metal_id > 0:
        item = generated[metal_id - 1]
        cup, stem = item['cup'], item['stem']
        pred = item.get('pred')
        cond_values = item.get('condition') or {}
        cond_title = '生成条件'

    view_cols = stx.columns([1, 1], vertical_alignment='top')
    with view_cols[0]:
        st.caption(f'{canvas_t} {ax_t} 透视')
        drr_images = render_drr(prl, canvas_t, ax, canvas, cup, stem)
        if drr_images:
            view_cols[0].image(drr_images[0][1])
    with view_cols[1]:
        extra = []
        if metal_id > 0:
            extra.append(CONDITION_MODE_LABELS.get(cond_values.get('mode'), ''))
            if cond_values.get('seed') is not None:
                extra.append(f'种子 {cond_values["seed"]}')
        extra.append('生成域分类器')
        st.caption(f'{cond_title} / ' + ' · '.join(part for part in extra if part))
        rows = _compare_rows(cond_values, pred)
        st.dataframe(
            {
                '参数': [row[0] for row in rows],
                cond_title: [row[1] for row in rows],
                '判别': [row[2] for row in rows],
                '置信度': [row[3] for row in rows],
            },
            hide_index=True,
            width='stretch',
        )
        if pred is None:
            st.caption('生成一次后会同时判别真实假体与生成假体。')
        else:
            if pred.get('stem_model', {}).get('top3'):
                top3 = '，'.join(f'{label or "未标注"} {prob:.1%}' for label, prob in pred['stem_model']['top3'])
                st.caption(f'型号 Top-3：{top3}')
            if (
                metal_id > 0
                and cond_values.get('stem_model')
                and pred['stem_model']['label']
                and cond_values['stem_model'] != pred['stem_model']['label']
            ):
                st.warning(f'指定型号为 {cond_values["stem_model"]}，判别结果为 {pred["stem_model"]["label"]}。骨骼约束可能已改变实际几何。')

    pack_head = stx.columns([1, 1, 6], vertical_alignment='bottom')
    show_slices = pack_head[0].checkbox('断层')
    cn = 10
    if show_slices:
        cn = pack_head[1].number_input('列数', 1, 100, 10, 1)
        slice_images = render_slices(prl, canvas_t, ax, canvas, cup, stem)
        with stx.expander(f'{canvas_t}{ax_t}断层', expanded=True):
            for i in range(0, len(slice_images), cn):
                slice_cols = st.columns(cn)
                for j in range(cn):
                    if i + j < len(slice_images):
                        caption, rgb = slice_images[i + j]
                        slice_cols[j].image(rgb, '{} = {}'.format('XYZ'[ax], caption))

    cols = stx.columns([2, 1, 1, 1, 1, 1, 1], vertical_alignment='bottom')

    options = ['术前图像', '术前骨骼模型', '假体距离场', '假体模型']
    selected = cols[0].multiselect('打包', options, options[-1], width='stretch')

    if cols[1].button('导出', width='stretch'):
        with cols[2].spinner('正在打包'):
            with tempfile.TemporaryDirectory() as tempdir:
                savedir = Path(tempdir) / '{}_{}'.format(prl, 'fake' if metal_id > 0 else 'true')
                i6_export(savedir, cup, stem, pre_path, pre_origin, pre_spacing, pre_direction)

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

            cols[2].download_button('下载', data=memory_file.getvalue(), file_name=f'{savedir.name}.zip', mime='application/zip')
