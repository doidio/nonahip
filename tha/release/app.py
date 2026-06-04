# uv run streamlit run release/app.py --server.port 8505 -- --config config.toml

import argparse
import copy
from pathlib import Path

import itk
import numpy as np
import plotly.express as px
import streamlit as st
import tomlkit
from infer import FEMORAL

st.set_page_config('Nonavox/THA', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonavox/THA 推理')


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


@st.cache_data(show_spinner='正在作图', show_time=True)
def gen_slice_images(prl, canvas_name, ax, k_min, k_max, _canvas, _cup, _stem):
    kn = k_max - k_min + 1
    slice_images = []
    for i in range(kn):
        k = k_max - i
        p = np.take(_canvas, k, axis=ax).transpose(1, 0)
        c = np.take(_cup, k, axis=ax).transpose(1, 0)
        s = np.take(_stem, k, axis=ax).transpose(1, 0)

        if ax in (0, 1):
            p, c, s = np.flipud(p), np.flipud(c), np.flipud(s)

        rgb = np.stack([p, p, p], axis=-1)

        c_mask = c > 0.0
        rgb[c_mask] = (1.0 - rgb[c_mask]) * 0.5 + np.array([0.5, 1.0, 0.5]) * 0.5

        s_mask = s > 0.0
        rgb[s_mask] = (1.0 - rgb[s_mask]) * 0.5 + np.array([0.0, 0.5, 1.0]) * 0.5

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        slice_images.append((k, rgb_uint8))
    return slice_images


if (it := st.session_state.get('init')) is None:
    gen_slice_images.clear()

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

    def format_func(_):
        rt = [f'{_}']

        spec = tests[_]['context']['femoral_spec']

        if len(spec[0]):
            rt.append(f'型号: {spec[0]}')

        if len(spec[1]):
            rt.append(f'规格: {spec[1]}')

        cup_outer = str(tests[_]['context'].get('cup_outer_best', tests[_]['context'].get('cup_outer', '')))

        if len(cup_outer):
            rt.append(f'杯径: {cup_outer}')

        liner_offset = str(tests[_]['context'].get('liner_offset_best', tests[_]['context'].get('liner_offset', '')))

        if len(liner_offset):
            rt.append(f'衬偏: {liner_offset}')

        head_outer = str(tests[_]['context'].get('head_outer', ''))

        if len(head_outer):
            rt.append(f'头径: {head_outer}')

        return ' '.join(rt)

    prl = st.selectbox('测试集', list(sorted(tests.keys())), format_func=format_func)
    it = tests[prl]

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

    st.code(tomlkit.dumps(it), 'toml')

else:
    cfg, tests = st.session_state['init']
    prl, pre, post_align_hip, post_align_femur, cup, stem = st.session_state['prl']
    it = tests[prl]

    with st.expander(prl, expanded=False):
        st.code(tomlkit.dumps(it), 'toml')

    cols = st.columns(3)

    canvas_dict = {'术前': pre, '术后对齐骨盆': post_align_hip, '术后对齐股骨': post_align_femur}
    canvas_name = cols[0].radio('底图', list(canvas_dict.keys()), horizontal=True)
    canvas = canvas_dict[canvas_name]

    ax = {'冠状面': 1, '矢状面': 0, '横断面': 2}
    ax = ax[cols[1].radio('方位', list(ax.keys()), horizontal=True)]

    cn = cols[2].number_input('列数', 1, 100, 10)

    axes = tuple(i for i in range(3) if i != ax)
    cup_ax = np.any(cup > 0.5, axis=axes)
    stem_ax = np.any(stem > 0.5, axis=axes)

    cup_indices = np.where(cup_ax)[0]
    stem_indices = np.where(stem_ax)[0]

    cup_min = cup_indices[0] if len(cup_indices) > 0 else 0
    cup_max = cup_indices[-1] if len(cup_indices) > 0 else 0
    stem_min = stem_indices[0] if len(stem_indices) > 0 else 0
    stem_max = stem_indices[-1] if len(stem_indices) > 0 else 0

    if len(cup_indices) > 0 and len(stem_indices) > 0:
        k_min, k_max = min(cup_min, stem_min), max(cup_max, stem_max)
    else:
        k_min, k_max = cup_min or stem_min, cup_max or stem_max

    slice_images = gen_slice_images(prl, canvas_name, ax, k_min, k_max, canvas, cup, stem)
    with st.expander('真值 Ground Truth'):
        kn = len(slice_images)
        for i in range(0, kn, cn):
            cols = st.columns(cn)
            for j in range(cn):
                if i + j < kn:
                    k, rgb = slice_images[i + j]
                    cols[j].image(rgb, '{} = {}'.format('XYZ'[ax], k))

    # femoral = {}
    # for orig_a in FEMORAL:
    #     a = '<空>' if len(orig_a) == 0 else orig_a
    #     femoral[a] = {}
    #     for orig_b in FEMORAL[orig_a]:
    #         b = '<空>' if len(orig_b) == 0 else orig_b
    #         femoral[a][b] = orig_b

    # spec_0 = st.radio('型号', list(femoral.keys()), horizontal=True)
    # sepc_1 = st.radio('规格', list(femoral[spec_0].keys()), horizontal=True)
