# uv run streamlit run b1_prepare_context.py --server.port 8501 -- --config config.toml

import argparse
import copy
import tempfile
from io import BytesIO
from pathlib import Path

import itk
import numpy as np
import streamlit as st
import tomlkit
import trimesh
import warp as wp
from b0_config import load_pairs
from b0_preload_prothesis import FEMORAL, HEAD_OFFSET
from define import ct_metal, ct_seg_femur_left, ct_seg_femur_right, ct_seg_hip_left, ct_seg_hip_right
from kernel import count_cup_head_3d, resample_cup_head
from PIL import Image, ImageDraw, ImageFont


@st.cache_resource(show_spinner=False)
def cache_load_pairs(config_file: str):
    cfg = Path(config_file)
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()
    return load_pairs(cfg, ['roi', 'context'])


save_key = 'head_center'

st.set_page_config('Nonahip', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonahip 全局标签录入')

# --- 第一阶段：初始化与数据列表加载 ---
if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args, _ = parser.parse_known_args()
        cfg, pairs = cache_load_pairs(args.config)

    st.session_state['init'] = cfg, pairs
    st.rerun()

# --- 第二阶段：病例选择界面 ---
elif (it := st.session_state.get('prl')) is None:
    cfg, pairs = st.session_state['init']

    # 统计配准进度
    dn = len([_ for _ in pairs if save_key in pairs[_]['context'] or pairs[_].get('excluded', False)])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud} 个样本', label_visibility='collapsed')

    # 自动跳转到下一个待配准的病例
    if st.button('下一个'):
        for prl in pairs:
            if save_key not in pairs[prl]['context'] and not pairs[prl].get('excluded', False):
                st.session_state['prl_input'] = prl
                break

    # 手动输入或选择病例 ID
    prl = st.text_input('PatientID_RL', key='prl_input')
    if prl in pairs:
        if st.button('确定'):
            st.session_state['prl'] = prl
            st.rerun()

        st.code(tomlkit.dumps(pairs[prl]), 'toml')

# --- 第三阶段：加载图像与分割计算 ROI ---
elif (it := st.session_state.get('preload')) is None:
    cfg, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    root = Path(cfg['dataset']['root'])

    seg_labels = [
        {'R': ct_seg_hip_right, 'L': ct_seg_hip_left}[rl],
        {'R': ct_seg_femur_right, 'L': ct_seg_femur_left}[rl],
    ]

    # 术后
    roi_boxes = []
    for part in ('hip', 'femur'):
        origin = np.array(pairs[prl]['roi']['post'][part]['origin'])
        spacing = np.array(pairs[prl]['roi']['post'][part]['spacing'])
        size = np.array(pairs[prl]['roi']['post'][part]['size'])
        roi_boxes.append([origin, origin + spacing * size])

    with tempfile.TemporaryDirectory() as tdir:
        with st.spinner(_ := '加载原图', show_time=True):  # noqa
            f = root / 'nii' / pid / pairs[prl]['pair']['post']

            image = itk.imread(f.as_posix(), itk.SS)

            size = np.array(itk.size(image), float)
            spacing = np.array(itk.spacing(image), float)
            origin = np.array(itk.origin(image), float)

            image = np.ascontiguousarray(itk.array_from_image(image).transpose(2, 1, 0))
            image_bg = float(np.min(image))

            volume = wp.Volume.load_from_numpy(image, bg_value=image_bg)

        with st.spinner(_ := '加载股骨', show_time=True):  # noqa
            f = root / 'pair' / pid / rl / 'post' / 'femur' / 'bone.stl'
            _ = trimesh.load_mesh(f.as_posix())
            _ = max(_.split(), key=lambda _: _.area)
            bone_mesh = _

    st.session_state['preload'] = image, volume, size, spacing, origin, image_bg, roi_boxes, bone_mesh
    st.rerun()

# --- 第四阶段：手动标注 ---
else:
    cfg, pairs = st.session_state['init']
    image, volume, size, spacing, origin, image_bg, roi_boxes, bone_mesh = st.session_state['preload']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    root = Path(cfg['dataset']['root'])

    saved = copy.deepcopy(pairs[prl]['context'])

    with st.expander(prl):
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

    roi_boxes = np.array(roi_boxes)

    cols = st.columns([2, 3, 3, 3])

    # 股骨柄型号规格
    with cols[0]:
        spec_0, spec_1 = saved.get('femoral_spec', ['', ''])

        if spec_0 not in FEMORAL:
            spec_0 = ''

        if spec_1 not in FEMORAL[spec_0]:
            spec_1 = ''

        options = list(FEMORAL.keys())
        spec_0 = st.selectbox('股骨柄型号', options, options.index(spec_0) if spec_0 in options else 0)

        if len(spec_0):
            options = FEMORAL[spec_0]
            spec_1 = st.selectbox('股骨柄规格', options, options.index(spec_1) if spec_1 in options else 0)
        else:
            spec_1 = ''

    # 股骨头偏距
    with cols[0]:
        head_offset = saved.get('head_offset', '')
        if head_offset not in HEAD_OFFSET:
            head_offset = ''
        head_offset = st.selectbox('股骨头偏距/颈长偏移 (mm)', HEAD_OFFSET, HEAD_OFFSET.index(head_offset))

    # 三视图
    view_size = cols[0].radio('视野范围 (mm)', [100, 200, 300], horizontal=True)
    view_window = cols[0].radio('窗', ['高亮', '假体', '骨骼'], horizontal=True)

    ct_highlight = ct_metal - 500.0  # 适当降低阈值以兼容较大球头内部存在的较暗伪影
    window = {'高亮': [ct_highlight, 0.0], '假体': [ct_highlight, 1000.0], '骨骼': [-100.0, 1000.0]}[view_window]

    cup_outer = int(saved.get('cup_outer_best', saved.get('cup_outer', 90)))
    cup_outer = cols[0].number_input('髋臼杯外径', 10, 90, cup_outer, 2, key='cup_outer')

    head_outer = int(saved.get('head_outer', 10))
    head_outer = cols[0].number_input('股骨头外径', 10, 90, head_outer, 2, key='head_outer')

    liner_slot = cols[0].empty()

    step = cols[0].radio('调节 (mm/deg)', _ := ['5', '25', '50', '0.5', '0.25', '0.25 + 边缘精修'], horizontal=True)

    if step in _[:-1]:
        step = float(step)
        shell_only = False
    else:
        step = 0.25
        shell_only = True

    # 初始估计球心
    v_max = bone_mesh.vertices[np.argmax(bone_mesh.vertices[:, 2])]
    head_center_default = v_max.copy()
    head_center_default += roi_boxes[1][0]
    head_center_default[2] -= 15

    head_center_default = np.array(head_center_default)
    head_center_default = np.round(head_center_default / 0.25) * 0.25
    head_center_default = head_center_default.tolist()

    if 'head_center' not in st.session_state:
        st.session_state['head_center'] = np.array(saved.get('head_center', head_center_default)).copy()

    cup_axis_default = np.array([(1 if rl == 'L' else -1) * np.sin(np.deg2rad(40)), 0, -np.cos(np.deg2rad(40))])
    cup_axis_default /= np.linalg.norm(cup_axis_default)

    if 'cup_axis' not in st.session_state:
        st.session_state['cup_axis'] = np.array(saved.get('cup_axis', cup_axis_default)).copy()

    img_slots = [cols[1 + _].container() for _ in range(3)]

    for i in range(3):
        col_l, col_m, col_r = cols[1 + i].columns(3, vertical_alignment='bottom')

        axes = [0, 1, 2]
        del axes[i]

        a, d = 'RAI'[axes[0]], 'LPS'[axes[0]]
        s, w = 'RAI'[axes[1]], 'LPS'[axes[1]]

        # +y↓: SSP -> IIP
        if i not in (0, 1):
            s, w = w, s

        if col_l.button('↺', key=f'↺_{i}', width='stretch'):
            v = np.array(st.session_state['cup_axis'])
            x, y = v[axes[0]], (-v[axes[1]] if i in (0, 1) else v[axes[1]])
            theta = np.deg2rad(step)
            cos, sin = np.cos(theta), np.sin(theta)
            v[axes[0]], y = x * cos + y * sin, -x * sin + y * cos
            v[axes[1]] = -y if i in (0, 1) else y
            st.session_state['cup_axis'] = v / np.linalg.norm(v)

        if col_r.button('↻', key=f'↻_{i}', width='stretch'):
            v = np.array(st.session_state['cup_axis'])
            x, y = v[axes[0]], (-v[axes[1]] if i in (0, 1) else v[axes[1]])
            theta = np.deg2rad(-step)
            cos, sin = np.cos(theta), np.sin(theta)
            v[axes[0]], y = x * cos + y * sin, -x * sin + y * cos
            v[axes[1]] = -y if i in (0, 1) else y
            st.session_state['cup_axis'] = v / np.linalg.norm(v)

        if col_l.button(f'{a}', key=f'{a}_{i}', width='stretch'):
            st.session_state['head_center'][axes[0]] += step * (1 if a in 'LPS' else -1)
        if col_r.button(f'{d}', key=f'{d}_{i}', width='stretch'):
            st.session_state['head_center'][axes[0]] += step * (1 if d in 'LPS' else -1)

        if col_m.button(f'{w}', key=f'{w}_{i}', width='stretch'):
            st.session_state['head_center'][axes[1]] += step * (1 if w in 'LPS' else -1)
        if col_m.button(f'{s}', key=f'{s}_{i}', width='stretch'):
            st.session_state['head_center'][axes[1]] += step * (1 if s in 'LPS' else -1)

    head_center = np.array(st.session_state['head_center'], float)
    cup_axis = np.array(st.session_state['cup_axis'], float)

    roi_spacing = 0.2
    roi_size = np.ceil((np.ones(3) * view_size) / roi_spacing).astype(int)

    def get_occupancy(hc, ca, lo):
        cc = hc - lo * ca
        o = cc - 0.5 * roi_size * roi_spacing

        counts: wp.array = wp.zeros(10, dtype=wp.int32)  # noqa
        wp.launch(
            count_cup_head_3d,
            tuple(roi_size),
            [
                volume.id,
                wp.vec3(origin),
                wp.vec3(spacing),
                ct_highlight,
                wp.vec3(o),
                roi_spacing,
                wp.vec3(cc),
                wp.vec3(ca),
                wp.vec3(hc),
                head_outer / 2.0,
                cup_outer / 2.0,
                counts,
                shell_only,
            ],
        )

        c = counts.numpy()
        head_roi_sum = float(c[0])
        head_metal_sum = float(c[1])
        out_head_roi_sum = float(c[2])
        out_head_metal_sum = float(c[3])

        cup_roi_sum = float(c[4])
        cup_metal_sum = float(c[5])
        out_cup_roi_sum = float(c[6])
        out_cup_metal_sum = float(c[7])

        liner_roi_sum = float(c[8])
        liner_metal_sum = float(c[9])

        h_occ = head_metal_sum / head_roi_sum if head_roi_sum > 0 else 0.0
        out_h_occ = out_head_metal_sum / out_head_roi_sum if out_head_roi_sum > 0 else 0.0

        c_occ = cup_metal_sum / cup_roi_sum if cup_roi_sum > 0 else 0.0
        out_c_occ = out_cup_metal_sum / out_cup_roi_sum if out_cup_roi_sum > 0 else 0.0

        l_occ = liner_metal_sum / liner_roi_sum if liner_roi_sum > 0 else 0.0

        return (h_occ - out_h_occ) if shell_only else h_occ, c_occ - out_c_occ, l_occ

    cols[0].caption('高亮占比（3D）')
    empty = cols[0].empty()

    if cols[0].button('自动调节', width='stretch'):
        # 斐波那契球面均匀采样方向
        samples = 26
        phi = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(samples)
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        sphere_directions = np.column_stack((x, y, z)).tolist()

        liner_offset_best: float = (
            st.session_state.get('liner_offset_best', saved.get('liner_offset', 0.0)) if step > 0.25 else saved.get('liner_offset', 0.0)
        )
        occ_max = get_occupancy(head_center, cup_axis, liner_offset_best)

        better = True
        while better:  # 位置和内衬偏心联合优化
            better = False

            # 1. 试探位置
            for offset in sphere_directions + [-cup_axis, cup_axis]:
                offset = np.array(offset, float)
                offset /= np.linalg.norm(offset)
                head_center_test = head_center + offset * step

                occ = get_occupancy(head_center_test, cup_axis, liner_offset_best)

                if occ_max[0] * 0.8 + occ_max[1] * 0.2 < occ[0] * 0.8 + occ[1] * 0.2:
                    occ_max = occ
                    head_center = head_center_test

                    empty.info(f'头 {occ_max[0] * 1e2:.3f}% 杯 {occ_max[1] * 1e2:.3f}% 衬 {occ_max[2] * 1e2:.3f}%')
                    better = True
                    break

            # 2. 试探内衬偏心 (仅在精修阶段)
            if step <= 0.25:
                for _ in range(0, int(6.0 // 0.25) + 1):
                    lo_test = _ * 0.25
                    if lo_test == liner_offset_best:
                        continue
                    occ = get_occupancy(head_center, cup_axis, lo_test)

                    if occ_max[0] * 0.2 + occ_max[1] * 0.8 < occ[0] * 0.2 + occ[1] * 0.8:
                        occ_max = occ
                        liner_offset_best = lo_test

                        empty.info(f'头 {occ_max[0] * 1e2:.3f}% 杯 {occ_max[1] * 1e2:.3f}% 衬 {occ_max[2] * 1e2:.3f}%')
                        better = True
                        break

        better = True
        while better:  # 朝向
            better = False
            for axis in [
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ]:
                k = np.array(axis, float)
                v = cup_axis.copy()
                theta = np.deg2rad(step)

                # 罗德里格旋转公式
                v = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
                v /= np.linalg.norm(v)

                occ = get_occupancy(head_center, v, liner_offset_best)

                if occ_max[0] * 0.2 + occ_max[1] * 0.8 < occ[0] * 0.2 + occ[1] * 0.8:
                    occ_max = occ
                    cup_axis = v.copy()

                    empty.info(f'头 {occ_max[0] * 1e2:.3f}% 杯 {occ_max[1] * 1e2:.3f}% 衬 {occ_max[2] * 1e2:.3f}%')
                    better = True
                    break

        empty.info(f'头 {occ_max[0] * 1e2:.3f}% 杯 {occ_max[1] * 1e2:.3f}% 衬 {occ_max[2] * 1e2:.3f}%')

        st.session_state['head_center'] = head_center.tolist()
        st.session_state['cup_axis'] = cup_axis.tolist()
        st.session_state['liner_offset_best'] = liner_offset_best

    _ = 'liner_offset_best'
    st.session_state[_] = st.session_state.get(_, saved.get(_, saved.get('liner_offset', 0.0)))

    liner_offset: float = liner_slot.number_input('内衬偏心距', 0.0, 6.0, step=0.25, format='%.2f', key='liner_offset_best')

    cup_center = head_center - liner_offset * cup_axis

    shell_only = False
    occ_max = get_occupancy(head_center, cup_axis, liner_offset)
    empty.info(f'头 {occ_max[0] * 1e2:.3f}% 杯 {occ_max[1] * 1e2:.3f}% 衬 {occ_max[2] * 1e2:.3f}%')

    ort = [
        ['S', 'I', 'A', 'P'],
        ['S', 'I', 'R', 'L'],
        ['A', 'P', 'R', 'L'],
    ]

    stack = []
    for i in range(3):
        axes = np.eye(3, dtype=float).tolist()
        del axes[i]
        axes = np.array(axes)

        shape = [*roi_size]
        del shape[i]

        roi_slice: wp.array = wp.zeros(shape, dtype=wp.vec3ub)  # type: ignore
        roi_origin = head_center - 0.5 * roi_size * roi_spacing * axes[0] - 0.5 * roi_size * roi_spacing * axes[1]

        wp.launch(
            resample_cup_head,
            roi_slice.shape,
            [
                volume.id,
                wp.vec3(origin),
                wp.vec3(spacing),
                *window,
                roi_slice,
                wp.vec3(roi_origin),
                roi_spacing,
                wp.vec3(axes[0]),
                wp.vec3(axes[1]),
                wp.vec3(cup_center),
                wp.vec3(cup_axis),
                wp.vec3(head_center),
                head_outer / 2.0,
                cup_outer / 2.0,
            ],
        )

        roi_slice: np.ndarray = roi_slice.numpy().transpose(1, 0, 2)

        # +y↓: SSP -> IIP
        if i in (0, 1):
            roi_slice = np.flipud(roi_slice)

        img = Image.fromarray(roi_slice)

        # orientation
        draw = ImageDraw.Draw(img)
        cw, ch = [_ / 2 for _ in img.size]

        fill = (255, 255, 255)
        try:
            font = ImageFont.truetype('timesbd.ttf', 20)
        except (OSError, Exception):
            font = ImageFont.load_default()

        draw.text((cw, 20), ort[i][0], fill, font, anchor='mm')
        draw.text((cw, ch * 2 - 20), ort[i][1], fill, font, anchor='mm')
        draw.text((20, ch), ort[i][2], fill, font, anchor='mm')
        draw.text((cw * 2 - 20, ch), ort[i][3], fill, font, anchor='mm')

        stack.append(np.array(img))

    images = []
    captions = ['矢状面 (Sagittal)', '冠状面 (Coronal)', '横断面 (Axial)']
    for _ in range(3):
        img_slots[_].image(stack[_], captions[_], width='stretch')

        _ = Image.fromarray(stack[_])
        buf = BytesIO()
        _.save(buf, format='PNG')
        images.append(buf.getvalue())

    save = {
        'femoral_spec': [spec_0, spec_1],
        'head_offset': head_offset,
        'cup_outer_best': cup_outer,
        'head_outer': head_outer,
        'head_center': head_center.tolist(),
        'cup_center': cup_center.tolist(),
        'cup_axis': cup_axis.tolist(),
        'occupancy': list(occ_max),
        'liner_material': '陶瓷' if occ_max[2] > 0.5 else '聚乙烯',
        'liner_offset_best': liner_offset,
    }
    if 'liner_offset' in saved:
        save['liner_offset'] = saved['liner_offset']

    # 提交结果
    with st.form('submit'):
        if pairs[prl].get('excluded', False):
            st.warning(f'已排除 {prl}')

        for k, v in save.items():
            if isinstance(v, dict) and k in saved and isinstance(saved[k], dict):
                saved[k].update(v)
            else:
                saved[k] = v
        st.code(tomlkit.dumps({'context': saved}), 'toml')

        if st.form_submit_button('提交（覆盖）' if save_key in saved else '提交'):
            # 更新内存中的总表
            pairs[prl]['context'] = saved

            f = root / 'pair' / pid / rl / 'context.toml'
            f.write_bytes(tomlkit.dumps(saved).encode('utf-8'))

            # 保留 init 状态，只清空当前会话的其他状态
            st.session_state.clear()
            st.session_state['init'] = cfg, pairs
            st.rerun()
