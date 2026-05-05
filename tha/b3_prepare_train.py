# uv run streamlit run b3_prepare_train.py --server.port 8503 -- --config config.toml

import argparse
import copy
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
import tomlkit
from b0_config import load_pairs
from b3_predict_train import preload


@st.cache_resource(show_spinner=False)
def cache_load_pairs(config_file: str):
    cfg = Path(config_file)
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()
    return load_pairs(cfg, ['roi', 'context', 'align', 'train'])


save_key = 'train_ready'

st.set_page_config('Nonavox/THA', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonavox/THA 训练集检视')

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args, _ = parser.parse_known_args()
        cfg, pairs = cache_load_pairs(args.config)

    st.session_state['init'] = cfg, pairs
    st.rerun()

elif (it := st.session_state.get('prl')) is None:
    cfg, pairs = st.session_state['init']

    dn = len([prl for prl in pairs if save_key in pairs[prl]['train'] or pairs[prl].get('excluded', False)])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud} 个样本', label_visibility='collapsed')

    if st.button('下一个'):
        for prl in pairs:
            if save_key not in pairs[prl]['train'] and not pairs[prl].get('excluded', False):
                st.session_state['prl_input'] = prl
                break

    prl = st.text_input('PatientID_RL', key='prl_input')
    if prl in pairs:
        if st.button('确定'):
            st.session_state['prl'] = prl
            st.rerun()
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

elif (it := st.session_state.get('preload')) is None:
    cfg, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    root = Path(cfg['dataset']['root'])

    with st.spinner('预载', show_time=True):
        preloaded = preload(cfg, pairs[prl])

    roi_origin, roi_spacing, images = preloaded

    # with tempfile.TemporaryDirectory() as tdir:
    #     for i, image in enumerate(images):
    #         f = Path(tdir) / 'image.nii.gz'
    #         image = itk.image_from_array(np.ascontiguousarray(image.transpose(2, 1, 0)))
    #         image.SetOrigin(roi_origin)
    #         image.SetSpacing(roi_spacing)
    #         itk.imwrite(image, f.as_posix())
    #         images[i] = f.read_bytes()

    st.session_state['preload'] = roi_origin, roi_spacing, images
    st.rerun()

else:
    cfg, pairs = st.session_state['init']
    roi_origin, roi_spacing, images = st.session_state['preload']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    root = Path(cfg['dataset']['root'])

    saved = copy.deepcopy(pairs[prl]['train'])

    cols = st.columns([1, 2])

    options = ['术前', '术后（配准髋臼侧)', '术后（配准股骨侧）', '术前（融合假体）', '髋臼杯', '股骨柄']
    select = cols[0].radio('类别', options, horizontal=True)
    image = dict(zip(options, images))[select]

    sub_cols = cols[0].columns([3, 3, 1, 1])

    options = ['横断位', '冠状位', '矢状位']
    select = sub_cols[0].radio('方位', options, horizontal=True)
    ort = dict(zip(options, (2, 1, 0)))[select]

    i_max = image.shape[ort] - 1
    if 'slice_i' not in st.session_state:
        st.session_state['slice_i'] = i_max // 2
    else:
        st.session_state['slice_i'] = int(np.clip(st.session_state['slice_i'], 0, i_max))

    target = st.session_state.get('target_slice_i', st.session_state['slice_i'])
    target = int(np.clip(target, 0, i_max))
    st.session_state['target_slice_i'] = target

    if st.session_state['slice_i'] != target:
        st.session_state['slice_i'] += 1 if target > st.session_state['slice_i'] else -1

    i = sub_cols[1].number_input(f'位置 (0 ~ {i_max})', 0, i_max, key='slice_i')

    if sub_cols[2].button('-10', width='stretch'):
        st.session_state['target_slice_i'] = max(0, i - 10)
        st.rerun()

    if sub_cols[3].button('+10', width='stretch'):
        st.session_state['target_slice_i'] = min(i_max, i + 10)
        st.rerun()

    image = np.take(image, i, axis=ort)
    image = image.transpose(1, 0)
    if ort in (0, 1):
        image = np.flipud(image)

    def on_select(*args, **kwargs):
        print(args, kwargs)

    fig = px.imshow(image, color_continuous_scale='gray', range_color=[-1, 1])
    fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=0))
    chart = cols[1].plotly_chart(fig, on_select='rerun', selection_mode='box')

    if i != st.session_state.get('target_slice_i', i):
        st.rerun()

    # 提交
    # with cols[0]:
    #     with st.form('submit'):
    #         if pairs[prl].get('excluded', False):
    #             st.warning(f'已排除 {prl}')

    #         for k, v in save.items():
    #             if isinstance(v, dict) and k in saved and isinstance(saved[k], dict):
    #                 saved[k].update(v)
    #             else:
    #                 saved[k] = v
    #         st.code(tomlkit.dumps({'align': saved}), 'toml')

    #         if st.form_submit_button('提交（覆盖）' if save_key in saved else '提交'):
    #             # 更新内存中的总表
    #             pairs[prl]['align'] = saved

    #             f = root / 'pair' / pid / rl / 'align.toml'
    #             f.write_bytes(tomlkit.dumps(saved).encode('utf-8'))

    #             st.session_state.clear()
    #             st.session_state['init'] = cfg, pairs
    #             st.rerun()
