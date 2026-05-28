# uv run streamlit run release/app.py --server.port 8505 -- --config config.toml

import argparse
import copy
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from infer import FEMORAL
import tomlkit


st.set_page_config('Nonavox/THA', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonavox/THA 推理')


@st.cache_resource(show_spinner=False)
def cache_load_pairs(config_file: str):
    cfg = Path(config_file)
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()

    root = Path(cfg['dataset']['root'])
    pair = root / 'pair'
    excluded = cfg['pairs']['excluded']

    categories = ['pair', 'roi', 'align', 'context']

    pairs = {}
    for pid in pair.iterdir():
        for rl in 'RL':
            prl = pid / rl

            if not (prl / 'pair.toml').exists():
                continue

            it = {}

            for category in set(['pair'] + categories):
                f = prl / f'{category}.toml'
                if f.exists():
                    data = tomlkit.loads(f.read_text('utf-8')).unwrap()
                    it[category] = data
                else:
                    it[category] = {}

            prl = '_'.join([pid.name, rl])
            if prl in excluded:
                it['excluded'] = True

            it['prl'] = prl
            pairs[prl] = it
    return cfg, pairs


if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args, _ = parser.parse_known_args()
        cfg, pairs = cache_load_pairs(args.config)

        trains, vals, tests = {}, {}, {}

        for prl in pairs:
            if prl in cfg['pairs']['excluded']:
                continue

            spec = pairs[prl]['context']['femoral_spec']
            spec = [spec[_] if len(spec[_]) else '<空>' for _ in range(2)]
            cup_outer = str(pairs[prl]['context'].get('cup_outer_best', pairs[prl]['context'].get('cup_outer', '')))
            cup_outer = cup_outer if len(cup_outer) else '<空>'
            liner_offset = str(pairs[prl]['context'].get('liner_offset_best', pairs[prl]['context'].get('liner_offset', '')))
            liner_offset = liner_offset if len(liner_offset) else '<空>'
            head_outer = str(pairs[prl]['context'].get('head_outer', ''))
            head_outer = head_outer if len(head_outer) else '<空>'

            if prl in cfg['test']:
                tests[prl] = pairs[prl]
            elif prl in cfg['val']:
                vals[prl] = pairs[prl]
            else:
                trains[prl] = pairs[prl]

    st.session_state['init'] = cfg, pairs, trains, vals, tests
    st.rerun()

elif (it := st.session_state.get('prl')) is None:
    cfg, pairs, trains, vals, tests = st.session_state['init']

    options = {'测试集': tests, '验证集': vals, '训练集': trains}
    ds: dict = options[st.radio('数据集', list(options.keys()), horizontal=True)]

    femoral = {}
    for a in FEMORAL:
        if len(a) == 0:
            a = '<空>'

        femoral[a] = {}

        for b in FEMORAL[a]:
            femoral[a][b] = FEMORAL

    spec_0 = st.radio('型号', list(FEMORAL.keys()), horizontal=True)
    sepc_1 = st.radio('规格', list(FEMORAL[spec_0]), horizontal=True)

    def format_func(_):
        spec = ds[_]['context']['femoral_spec']
        spec = [spec[_] if len(spec[_]) else '<空>' for _ in range(2)]
        cup_outer = str(ds[_]['context'].get('cup_outer_best', ds[_]['context'].get('cup_outer', '')))
        cup_outer = cup_outer if len(cup_outer) else '<空>'
        liner_offset = str(ds[_]['context'].get('liner_offset_best', ds[_]['context'].get('liner_offset', '')))
        liner_offset = liner_offset if len(liner_offset) else '<空>'
        head_outer = str(ds[_]['context'].get('head_outer', ''))
        head_outer = head_outer if len(head_outer) else '<空>'

        return f'索引: {_} 型规: {spec[0]} {spec[1]} 杯径: {cup_outer} 头径: {head_outer} 衬偏: {liner_offset}'

    prl = st.selectbox('数据', list(sorted(ds.keys())), format_func=format_func)
