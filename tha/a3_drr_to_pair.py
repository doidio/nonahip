# uv run streamlit run a3_drr_to_pair.py --server.port 8500 -- --config config.toml

import argparse
import locale
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tomlkit
from PIL import Image
from minio import Minio, S3Error

locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

cfg_path = Path(args.config)
cfg = tomlkit.loads(cfg_path.read_text('utf-8')).unwrap()
client = Minio(**cfg['minio']['client'])

st.set_page_config('Nonavox/THA', initial_sidebar_state='collapsed')
st.markdown('### Nonavox/THA 术前术后配对')

if (it := st.session_state.get('ud')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        dn = [_.object_name[:-1] for _ in client.list_objects('pair')]
        ud = [_.object_name[:-1] for _ in client.list_objects('drr') if _.object_name[:-1] not in dn]

    st.session_state['dn'] = dn
    st.session_state['ud'] = ud
    st.rerun()

elif (it := st.session_state.get('pid')) is None:
    dn = st.session_state['dn']
    ud = st.session_state['ud']

    st.progress(_ := len(dn) / (len(dn) + len(ud)), text=f'{100 * _:.2f}%')
    st.metric(f'progress', f'{len(dn)} / {len(dn) + len(ud)} 个患者', label_visibility='collapsed')

    if len(ud) > 0 and st.button('下一个'):
        with st.spinner('下一个', show_time=True):  # noqa
            ud = st.session_state['pid_input'] = ud[0]

    pid = st.text_input('PatientID', key='pid_input')

    if '_' in pid:
        pid = pid.split('_')[0]

    if len(pid):
        pairs = {}
        for _ in client.list_objects('pair', pid, recursive=True):
            if not _.object_name.endswith('.nii.gz'):
                continue

            pid, rl, op, nii = _.object_name.split('/')
            prl = f'{pid}_{rl}'
            if prl not in pairs:
                pairs[prl] = {'prl': prl}
            pairs[prl][op] = f'{pid}/{nii}'

        if len(pairs):
            st.code(tomlkit.dumps(pairs), 'toml')

        if st.button('确定'):
            series = []

            for _ in client.list_objects('drr', pid + '/'):
                if not _.is_dir:
                    continue

                object_name = _.object_name[:-1]

                try:
                    client.stat_object('drr', f'{object_name}/invalid')
                    continue
                except S3Error:
                    pass

                with tempfile.TemporaryDirectory() as tdir:
                    f = Path(tdir) / 'info.toml'
                    client.fget_object('drr', f'{object_name}/{f.name}', f.as_posix())
                    info = tomlkit.loads(f.read_text('utf-8')).unwrap()

                    if any(
                        (
                            'DERIVED' in info['dicom']['ImageType'],
                            'SECONDARY' in info['dicom']['ImageType'],
                        )
                    ):
                        continue

                    f = Path(tdir) / 'axial.png'
                    client.fget_object('drr', f'{object_name}/{f.name}', f.as_posix())
                    axial = np.asarray(Image.open(f.as_posix()))

                    f = Path(tdir) / 'coronal.png'
                    client.fget_object('drr', f'{object_name}/{f.name}', f.as_posix())
                    coronal = np.asarray(Image.open(f.as_posix()))

                    StudyDate, StudyTime = info['dicom']['StudyDate'], info['dicom']['StudyTime']
                    dt = datetime(
                        year=int(StudyDate[0:4]),
                        month=int(StudyDate[4:6]),
                        day=int(StudyDate[6:8]),
                        hour=int(StudyTime[0:2]),
                        minute=int(StudyTime[2:4]),
                        second=int(StudyTime[4:6]),
                        microsecond=int(StudyTime[7:13]) if len(StudyTime) >= 13 else 0,
                    )
                    series.append([dt, info, axial, coronal, object_name])

            series = sorted(series, key=lambda _: _[0])

            st.session_state['pid'] = pid
            st.session_state['pairs'] = pairs
            st.session_state['series'] = series
            st.rerun()
else:
    dn = st.session_state['dn']
    ud = st.session_state['ud']

    pid = st.session_state['pid']
    pairs = st.session_state['pairs']
    series = st.session_state['series']

    R0 = st.session_state.get('R0')
    R1 = st.session_state.get('R1')
    L0 = st.session_state.get('L0')
    L1 = st.session_state.get('L1')

    st.progress(_ := len(dn) / (len(dn) + len(ud)), text=f'{100 * _:.2f}%')
    st.metric(f'PatientID {pid}', f'{len(dn)} / {len(dn) + len(ud)}')

    options = {f'[{i}]': i for i, t in enumerate(series)}
    if len(pairs):
        st.code(tomlkit.dumps(pairs), 'toml')

        ls = [_[-1] for _ in series]
        for prl in pairs:
            pid, rl = prl.split('_')
            pre = ls.index(pairs[prl]['pre'])
            post = ls.index(pairs[prl]['post'])
            pre, post = [f'[{_}]' for _ in (pre, post)]
            if pre in options and post in options and f'select_{rl}' not in st.session_state:
                st.session_state[f'select_{rl}'] = (pre, post)

    with st.form('submit'):
        rl = [[], []]
        for i, c in enumerate(st.columns(2)):
            rl[i] = c.multiselect(['右侧', '左侧'][i], options.keys(), key=f'select_' + 'RL'[i])

        n = set(len(_) for _ in rl)

        try:
            client.stat_object('pair', '/'.join([pid, 'pair.done']))
            _ = '提交（覆盖）'
        except S3Error:
            _ = '提交'

        if st.form_submit_button(_):
            if len(n - {0, 2}) > 0:
                st.error('选择数量错误，只能双选或不选')
            else:
                for _ in client.list_objects('pair', pid, recursive=True):
                    client.remove_object('pair', _.object_name)

                for i, c in enumerate(rl):
                    if len(c) == 2:
                        pre, post = [series[options[rl[i][_]]][-1] for _ in range(2)]
                        pre, post = [_.split('/')[-1] for _ in (pre, post)]
                        client.put_object('pair', '/'.join([pid, 'RL'[i], 'pre', pre]), BytesIO(b''), 0)
                        client.put_object('pair', '/'.join([pid, 'RL'[i], 'post', post]), BytesIO(b''), 0)
                client.put_object('pair', '/'.join([pid, 'pair.done']), BytesIO(b''), 0)

                st.session_state.clear()
                st.rerun()

    for i in options.values():
        t = series[i]
        with st.expander(f'[{i}] {t[0]}', expanded=True):
            st.caption(t[-1].split('/')[1])
            col1, col2 = st.columns(2)
            with col1:
                st.image(t[3])
            with col2:
                st.image(t[2])

            st.code(tomlkit.dumps(t[1]), 'toml')
