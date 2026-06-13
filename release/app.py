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
    CUP_OUTER,
    FEMORAL,
    HEAD_OFFSET,
    HEAD_OUTER,
    LINER_OFFSET,
    i1_load_models,
    i2_context_embed,
    i3_pre_encode,
    i4_rflow_sample,
    i5_metal_decode,
    i6_export,
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
    label = []

    spec = it['context']['femoral_spec']

    if spec[0] is not None and len(str(spec[0])):
        label.append(f'柄型号 {str(spec[0])}')

    if spec[1] is not None and len(str(spec[1])):
        label.append(f'柄规格 {str(spec[1])}')

    cup_outer = it['context'].get('cup_outer_best', it['context'].get('cup_outer'))

    if cup_outer is not None and len(str(cup_outer)):
        label.append(f'杯直径 {str(cup_outer)}')

    liner_offset = it['context'].get('liner_offset_best', it['context'].get('liner_offset'))

    if liner_offset is not None and len(str(liner_offset)):
        label.append(f'衬偏心 {str(liner_offset)}')

    head_outer = it['context'].get('head_outer')

    if head_outer is not None and len(str(head_outer)):
        label.append(f'头直径 {str(head_outer)}')

    seed = it.get('seed')

    if seed is not None and len(str(seed)):
        label.append(f'种子 {str(seed)}')

    return ' '.join(label)


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

    top_cols = cols[1].columns([1, 1, 1, 1, 1, 1], vertical_alignment='bottom')
    sub_cols = cols[1].columns([4, 1, 1])

    stx = st.container()
    log = st.expander('日志', expanded=True)

    canvas = {'术前': pre, '术后对齐骨盆': post_align_hip, '术后对齐股骨': post_align_femur}
    canvas_t = cols[0].radio('空间', list(canvas.keys()), horizontal=True)
    canvas = canvas[canvas_t]

    ax = {'正位': 1, '侧位': 0, '轴位': 2}
    ax_t = cols[0].radio('方位', list(ax.keys()), horizontal=True)
    ax = ax[ax_t]

    render_t = cols[0].radio('渲染', ['透视', '断层', '三维（暂不支持）'], horizontal=True)

    cn = cols[0].number_input('列数', 1, 100, 10, 1, width=200)

    is_cond = False

    if sub_cols[0].checkbox('股骨柄型号'):
        is_cond = True
        stem_brand = sub_cols[0].radio('型号', [_ for _ in FEMORAL.keys() if len(_)], horizontal=True, label_visibility='collapsed')

        if sub_cols[0].checkbox('股骨柄规格'):
            stem_size = sub_cols[0].radio('规格', [_ for _ in FEMORAL[stem_brand] if len(_)], horizontal=True, label_visibility='collapsed')
        else:
            stem_size = None
    else:
        stem_brand, stem_size = None, None

    if sub_cols[1].checkbox('杯直径'):
        is_cond = True
        cup_outer = sub_cols[1].number_input('杯直径', CUP_OUTER[0], CUP_OUTER[1], CUP_OUTER[2], CUP_OUTER[3], label_visibility='collapsed')
    else:
        cup_outer = None

    if sub_cols[1].checkbox('头直径'):
        is_cond = True
        head_outer = sub_cols[1].number_input('头直径', HEAD_OUTER[0], HEAD_OUTER[1], HEAD_OUTER[2], HEAD_OUTER[3], label_visibility='collapsed')
    else:
        head_outer = None

    if sub_cols[2].checkbox('头偏距'):
        is_cond = True
        head_offset = sub_cols[2].number_input('头偏距', HEAD_OFFSET[0], HEAD_OFFSET[1], HEAD_OFFSET[2], HEAD_OFFSET[3], label_visibility='collapsed')
    else:
        head_offset = None

    if sub_cols[2].checkbox('衬偏心'):
        is_cond = True
        liner_offset = sub_cols[2].number_input(
            '衬偏心', LINER_OFFSET[0], LINER_OFFSET[1], LINER_OFFSET[2], LINER_OFFSET[3], label_visibility='collapsed'
        )
    else:
        liner_offset = None

    if top_cols[3].checkbox('可复现'):
        seed = top_cols[2].number_input('复现种子', 0, None, 42, 1)
        samples = 1
    else:
        seed = None
        samples = top_cols[2].number_input('采样数量', 1, 100, 10, 1)

    timestemps = top_cols[4].number_input('采样步数 (Timesteps)', 1, 50, 5, 1)
    cf_guidance = top_cols[5].number_input('无分类器引导 (Classifier-Free Guidance)', 0.0, 9.0, 1.0, 1.0)

    if top_cols[1].button('清空'):
        del st.session_state['generated']
        st.rerun()

    if top_cols[0].button('条件生成' if is_cond else '无条件生成', width='stretch'):
        desc = it_desc({
            'prl': prl,
            'context': {
                'femoral_spec': [stem_brand, stem_size],
                'cup_outer': cup_outer,
                'head_outer': head_outer,
                'head_offset': head_offset,
                'liner_offset': liner_offset,
            },
            'seed': seed,
        })

        bar = stx.progress(0.0, '载入模型')
        vae_pre, vae_metal, (rflow, context_embedder) = i1_load_models(printf=lambda *args: log.caption('\t'.join(str(_) for _ in args)))

        bar.progress(0.2, '编码全局标签')
        context_emb, context_emb_uncond = i2_context_embed(context_embedder, stem_brand, stem_size, cup_outer, head_outer, head_offset, liner_offset)

        bar.progress(0.3, '编码术前图像')
        pre_encoded, *_ = i3_pre_encode(pre_path, *vae_pre)

        metal_latent = []
        for _ in range(samples):
            bar.progress(0.3 + 0.2 * (_ + 1) / samples, f'采样假体 {_ + 1} / {samples}')
            metal_latent.append(i4_rflow_sample(rflow, context_emb, context_emb_uncond, pre_encoded, seed, timestemps, cf_guidance))

        metal_tsdf = []
        for _ in range(samples):
            bar.progress(0.5 + 0.2 * (_ + 1) / samples, f'解码假体 {_ + 1} / {samples}')
            metal_tsdf.append(i5_metal_decode(metal_latent[_], pre_size, *vae_metal))

        bar.empty()

        del vae_pre, vae_metal, rflow, context_embedder
        del pre_encoded, metal_latent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        st.session_state['generated'].extend([(_, desc) for _ in metal_tsdf])

    def format_func(i):
        if i > 0:
            (_, _), desc = generated[i - 1]
            return f'预测 {i} {desc} '
        else:
            return '真实'

    generated = st.session_state.get('generated', [])
    metal_id = stx.radio('假体', range(len(generated) + 1), horizontal=True, format_func=format_func)

    if metal_id > 0:
        (cup, stem), desc = generated[metal_id - 1]

    if render_t == '透视':
        images = render_drr(prl, canvas_t, ax, canvas, cup, stem)
    elif render_t == '断层':
        images = render_slices(prl, canvas_t, ax, canvas, cup, stem)
    elif render_t == '三维':
        images = []
    else:
        images = []

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

    with stx.expander(f'{canvas_t}{ax_t}{render_t}', expanded=True):
        for i in range(0, len(images), cn):
            cols = st.columns(cn)
            for j in range(cn):
                if i + j < len(images):
                    caption, rgb = images[i + j]
                    cols[j].image(rgb, '{} = {}'.format('XYZ'[ax], caption) if render_t == '断层' else None)
