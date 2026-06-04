# uv run streamlit run release/app.py --server.port 8505 -- --config config.toml

import argparse
from pathlib import Path
from typing import Literal

import itk
import numpy as np
import streamlit as st
import tomlkit
from infer import CUP_OUTER, FEMORAL, HEAD_OFFSET, HEAD_OUTER, LINER_OFFSET

st.set_page_config('Nonavox/THA', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonavox/THA 假体预测生成')


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
    rt = [it['prl']]

    spec = it['context']['femoral_spec']

    if len(spec[0]):
        rt.append(f'型号: {spec[0]}')

    if len(spec[1]):
        rt.append(f'规格: {spec[1]}')

    cup_outer = str(it['context'].get('cup_outer_best', it['context'].get('cup_outer', '')))

    if len(cup_outer):
        rt.append(f'杯径: {cup_outer}')

    liner_offset = str(it['context'].get('liner_offset_best', it['context'].get('liner_offset', '')))

    if len(liner_offset):
        rt.append(f'衬偏: {liner_offset}')

    head_outer = str(it['context'].get('head_outer', ''))

    if len(head_outer):
        rt.append(f'头径: {head_outer}')

    return ' '.join(rt)


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

    prl = st.selectbox('测试集', list(sorted(tests.keys())), format_func=lambda _: it_desc(tests[_]))

    dataset = Path(cfg['train']['root']) / 'dataset'

    if st.button('载入'):
        with st.spinner('正在载入', show_time=True):
            pre = dataset / 'pre' / f'{prl}.nii.gz'
            post_align_hip = dataset / 'post_align_hip' / f'{prl}.nii.gz'
            post_align_femur = dataset / 'post_align_femur' / f'{prl}.nii.gz'
            cup = dataset / 'metal' / f'{prl}_cup.nii.gz'
            stem = dataset / 'metal' / f'{prl}_stem.nii.gz'

            images = []
            for i, f in enumerate((pre, post_align_hip, post_align_femur, cup, stem)):
                img = itk.imread(f.as_posix())
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
    it = tests[prl]
    desc = it_desc(it)

    with st.expander(f'{desc}', expanded=False):
        st.code(tomlkit.dumps(it), 'toml')

    cols = st.columns([2, 4, 1, 1])

    canvas = {'术前': pre, '术后对齐骨盆': post_align_hip, '术后对齐股骨': post_align_femur}
    canvas_t = cols[0].radio('空间', list(canvas.keys()), horizontal=True)
    canvas = canvas[canvas_t]

    ax = {'正位': 1, '侧位': 0, '轴位': 2}
    ax_t = cols[0].radio('方位', list(ax.keys()), horizontal=True)
    ax = ax[ax_t]

    render_t = cols[0].radio('渲染', ['透视', '断层', '三维'], horizontal=True)

    metal = cols[0].radio('假体', ['真实术后', '预测生成'], horizontal=True)

    cn = cols[0].number_input('列数', 1, 100, 10, 1, width=200)

    if cols[1].checkbox('型号'):
        spec_0 = cols[1].radio('型号', [_ for _ in FEMORAL.keys() if len(_)], horizontal=True, label_visibility='collapsed')

        if cols[1].checkbox('规格'):
            spec_1 = cols[1].radio('规格', [_ for _ in FEMORAL[spec_0] if len(_)], horizontal=True, label_visibility='collapsed')
        else:
            spec_1 = None
    else:
        spec_0, spec_1 = None, None

    if cols[2].checkbox('杯径'):
        cup_outer = cols[2].number_input('杯径', CUP_OUTER[0], CUP_OUTER[1], CUP_OUTER[2], CUP_OUTER[3])
    else:
        cup_outer = None

    if cols[2].checkbox('头径'):
        head_outer = cols[2].number_input('头径', HEAD_OUTER[0], HEAD_OUTER[1], HEAD_OUTER[2], HEAD_OUTER[3])
    else:
        head_outer = None

    if cols[3].checkbox('头偏'):
        head_offset = cols[3].number_input('头偏', HEAD_OFFSET[0], HEAD_OFFSET[1], HEAD_OFFSET[2], HEAD_OFFSET[3])
    else:
        head_offset = None

    if cols[3].checkbox('衬偏'):
        liner_offset = cols[3].number_input('衬偏', LINER_OFFSET[0], LINER_OFFSET[1], LINER_OFFSET[2], LINER_OFFSET[3])
    else:
        liner_offset = None

    cols = st.columns([2, 2, 1, 1, 1, 1], vertical_alignment='bottom')

    if cols[2].checkbox('可复现'):
        seed = cols[3].number_input('随机种子', 0, None, 42, 1)
        instances = 1
    else:
        seed = None
        instances = cols[3].number_input('随机数量', 1, 100, 10, 1)

    cfg_val = cols[4].number_input('CFG', 0.0, 9.0, 1.0, 1.0)
    ts = cols[5].number_input('Timesteps', 1, 50, 5, 1)

    if cols[1].button('生成', width='stretch'):
        cond = Path(cfg['train']['root']) / 'dataset' / 'pre' / f'{prl}.nii.gz'

    if render_t == '透视':
        images = render_drr(prl, canvas_t, ax, canvas, cup, stem)
    elif render_t == '断层':
        images = render_slices(prl, canvas_t, ax, canvas, cup, stem)
    elif render_t == '三维':
        images = []
    else:
        images = []

    with st.expander(f'{canvas_t}{ax_t}{render_t}{metal}假体', expanded=True):
        for i in range(0, len(images), cn):
            cols = st.columns(cn)
            for j in range(cn):
                if i + j < len(images):
                    caption, rgb = images[i + j]
                    cols[j].image(rgb, '{} = {}'.format('XYZ'[ax], caption) if render_t == '断层' else None)
