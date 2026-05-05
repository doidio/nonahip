# uv run streamlit run b2_prepare_align.py --server.port 8502 -- --config config.toml

import argparse
import copy
from pathlib import Path

import numpy as np
import pyvista as pv
import streamlit as st
import tomlkit
import trimesh
import warp as wp
from b0_config import load_pairs
from kernel import compute_sdf, icp


@st.cache_resource(show_spinner=False)
def cache_load_pairs(config_file: str):
    cfg = Path(config_file)
    cfg = tomlkit.loads(cfg.read_text('utf-8')).unwrap()
    return load_pairs(cfg, ['roi', 'align'])


save_key = 'hip_align'

st.set_page_config('Nonavox/THA', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonavox/THA 术前术后配准')

# --- 第一阶段：初始化与数据列表加载 ---
if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):
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
    dn = len([_ for _ in pairs if save_key in pairs[_]['align'] or pairs[_].get('excluded', False)])
    ud = len(pairs) - dn

    st.progress(_ := dn / (dn + ud), text=f'{100 * _:.2f}%')
    st.metric('progress', f'{dn} / {dn + ud} 个样本', label_visibility='collapsed')

    # 自动跳转到下一个待配准的病例
    if st.button('下一个'):
        for prl in pairs:
            if save_key not in pairs[prl]['align'] and not pairs[prl].get('excluded', False):
                st.session_state['prl_input'] = prl
                break

    # 手动输入或选择病例 ID
    prl = st.text_input('PatientID_RL', key='prl_input')
    if prl in pairs:
        if st.button('确定'):
            st.session_state['prl'] = prl
            st.rerun()
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

# --- 第三阶段：ROI 提取与三维重建 ---
elif (it := st.session_state.get('preload')) is None:
    cfg, pairs = st.session_state['init']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')

    root = Path(cfg['dataset']['root'])

    metal_meshes = {'femur': [], 'hip': []}
    bone_meshes = {'femur': [], 'hip': []}

    # 分别处理术前(op=0)和术后(op=1)数据
    for op in ('pre', 'post'):
        for part in ('femur', 'hip'):
            with st.spinner(_ := f'加载 {op} {part}', show_time=True):
                f = root / 'pair' / pid / rl / op / part / 'bone.stl'
                if not f.exists():
                    st.error(f'{op} {part} bone.stl 不存在')
                    st.stop()

                mesh = trimesh.load_mesh(f.as_posix())
                if not mesh.is_empty:
                    mesh = list(
                        sorted(
                            mesh.split(only_watertight=False),
                            key=lambda _: np.linalg.norm(_.bounds[1] - _.bounds[0]),
                            reverse=True,
                        )
                    )[0]
                else:
                    st.error(f'{op} {part} bone.stl 空网格体')
                    st.stop()
                bone_meshes[part].append(mesh)

                if op == 'post':
                    f = root / 'pair' / pid / rl / op / part / 'metal.stl'
                    if f.exists():
                        mesh = trimesh.load_mesh(f.as_posix())
                        metal_meshes[part].append(mesh)
                    else:
                        metal_meshes[part].append(None)
                else:
                    metal_meshes[part].append(None)

    st.session_state['preload'] = metal_meshes, bone_meshes
    st.rerun()

# --- 第四阶段：交互式配准与结果确认 ---
else:
    cfg, pairs = st.session_state['init']
    metal_meshes, bone_meshes = st.session_state['preload']
    prl = st.session_state['prl']
    pid, rl = prl.split('_')
    root = Path(cfg['dataset']['root'])

    saved = copy.deepcopy(pairs[prl]['align'])

    with st.expander(prl):
        st.code(tomlkit.dumps(pairs[prl]), 'toml')

    cols = st.columns(3)

    save = {'femur': {}, 'hip': {}}

    for part_id, part in enumerate(('hip', 'femur')):
        part_name = ['髋骨', '股骨'][part_id]

        with cols[0]:
            sizes = [np.array(pairs[prl]['roi'][_][part]['size']) for _ in ('pre', 'post')]
            spacings = [np.array(pairs[prl]['roi'][_][part]['spacing']) for _ in ('pre', 'post')]
            origins = [np.array(pairs[prl]['roi'][_][part]['origin']) for _ in ('pre', 'post')]

            post_mesh: trimesh.Trimesh = bone_meshes[part][1].copy()

            zl = [round(_.bounds[1][2] - _.bounds[0][2]) for _ in bone_meshes[part]]

            post_mesh_outlier: trimesh.Trimesh | None
            if part in ('femur',):
                _ = saved.get(part, {}).get('d_proximal', min(zl[1], 15))
                d_proximal: int = st.number_input(
                    f'{part_name}近端截除（0 ~ {zl[1]:.0f} mm）',
                    0,
                    zl[1],
                    _,
                    step=5,
                    key=f'{part}_d_proximal',
                    help='截除术后比术前多余的近端特征，或截除术后到大粗隆顶端',
                )

                # 根据 Z 轴高度裁剪术后网格，只保留用于配准的稳定骨干部分
                if d_proximal > 0 or zl[0] < zl[1]:
                    z_max = post_mesh.bounds[1][2] - d_proximal
                    z_min = z_max - zl[0]

                    z = post_mesh.vertices[:, 2]
                    mask = (z_min <= z) & (z <= z_max)
                    post_mesh.update_faces(np.all(mask[post_mesh.faces], axis=1))
                    post_mesh.remove_unreferenced_vertices()

                    # 记录被裁掉的部分以便在 3D 中对比显示
                    mask = ~mask
                    _outlier = bone_meshes[part][1].copy()
                    _outlier.update_faces(np.all(mask[_outlier.faces], axis=1))
                    _outlier.remove_unreferenced_vertices()
                    post_mesh_outlier = _outlier

                    if post_mesh.is_empty:
                        st.error('近端裁剪过多')
                        st.stop()
                else:
                    post_mesh_outlier = None
            else:
                d_proximal = 0
                post_mesh_outlier = None

            # 采样点纵向范围
            zl.append(round(post_mesh.bounds[1][2] - post_mesh.bounds[0][2]))
            _min, _max = int(d_proximal), int(d_proximal) + int(min(zl[0], zl[2]))

            _def = saved.get(part, {}).get('d_sample_range', (_min, _max))
            _def = (max(_min, min(int(_def[0]), _max)), max(_min, min(int(_def[1]), _max)))

            if _def[0] > _def[1]:
                _def = (_min, _max)

            d_sample_range = st.slider(
                f'{part_name}采样点范围（{_min} ~ {_max} mm）',
                _min,
                _max,
                _def,
                step=1,
                help='近端 ~ 远端',
                key=f'{part}_d_sample_range',
            )

            # 避开金属假体的距离
            _ = saved.get(part, {}).get('d_metal', 5 if part in ('femur',) else 15)
            d_metal: int = st.number_input(
                f'{part_name}采样点远离金属（0 ~ 50 mm）', 0, 50, _, step=5, key=f'{part}_d_metal'
            )

            with st.spinner(_ := '采样', show_time=True):
                max_dist = float(np.linalg.norm(sizes[1] * spacings[1]))
                d = wp.full((len(post_mesh.vertices),), max_dist, float)

                if metal_meshes[part][1] is not None:
                    # 使用 Warp 计算术后骨骼网格顶点到金属假体网格的距离 (SDF)
                    metal = wp.Mesh(
                        wp.array(metal_meshes[part][1].vertices, wp.vec3),
                        wp.array(metal_meshes[part][1].faces.flatten(), wp.int32),
                    )

                    wp.launch(
                        compute_sdf,
                        d.shape,
                        [
                            wp.uint64(metal.id),
                            wp.array(post_mesh.vertices, wp.vec3),
                            d,
                            max_dist,
                        ],
                    )

                d = d.numpy()

                if part == 'femur':
                    z0 = post_mesh.bounds[1][2] - post_mesh.vertices[:, 2]
                    z_mask = (d_sample_range[0] - d_proximal <= z0) & (z0 <= d_sample_range[1] - d_proximal)
                else:
                    z0 = post_mesh.vertices[:, 2] - post_mesh.bounds[0][2]
                    z_mask = (d_sample_range[0] <= z0) & (z0 <= d_sample_range[1])

                # 权重计算：距离金属越远、在采样范围内的顶点权重越高
                _ = d - d_metal
                _ = np.clip(_, 0, max(d_metal, 1e-6))
                _ *= z_mask

                if (n := min(int(np.sum(_ > 0)), 10000)) < 100:
                    st.error(f'采样点过少 {n}')
                    st.stop()
                elif n < 10000:
                    st.warning(f'采样点较少 {n}')

                # 随机采样 10000 个顶点用于 ICP 配准
                _ = _ / _.sum()
                _ = np.random.choice(len(post_mesh.vertices), size=n, replace=False, p=_)
                vertices = post_mesh.vertices[_]

            with st.spinner(_ := '配准', show_time=True):
                # 初始对齐：将术后网格的顶部对齐到术前网格的顶部（Z 轴平移）
                matrix = np.identity(4)
                if part == 'femur':
                    matrix[2, 3] = bone_meshes[part][0].bounds[1][2] - post_mesh.bounds[1][2]
                else:
                    matrix[2, 3] = bone_meshes[part][0].bounds[0][2] - post_mesh.bounds[0][2]

                # 执行 ICP (Iterative Closest Point) 算法精细对齐
                matrix, _, mse, iters = icp(
                    vertices,
                    bone_meshes[part][0],
                    matrix,
                    1e-5,
                    2000,
                    **dict(reflection=False, scale=False),
                )

            # 核心逻辑：计算从术后“原始图像坐标系”到术前“原始图像坐标系”的全局变换矩阵 (g_matrix)
            # ICP 得到的是 ROI 局部坐标系下的变换，需要结合 ROI 在原图中的偏移量进行还原
            offset = [origins[_] for _ in range(2)]

            pre = np.identity(4)
            pre[:3, 3] = offset[0]

            post_inv = np.identity(4)
            post_inv[:3, 3] = -offset[1]

            g_matrix = pre @ matrix @ post_inv

            xform = np.array(wp.transform_from_matrix(wp.mat44(g_matrix)), dtype=float).tolist()  # type: ignore

            save[f'{part}_align'] = xform

            if part in ('femur',):
                save[part]['d_proximal'] = d_proximal

            save[part]['d_sample_range'] = d_sample_range
            save[part]['d_metal'] = d_metal
            save[part]['post_points'] = len(vertices)
            save[part]['iterations'] = int(iters)
            save[part]['mse'] = float(mse)

        with cols[1 + part_id]:
            with st.spinner(_ := '场景', show_time=True):
                # 使用 PyVista 构建三维场景
                b = np.array(post_mesh.bounds)
                b[1][2] += d_proximal

                x, y, z = [b[1][_] - b[0][_] for _ in (0, 1, 2)]
                wx, wy, h = [round(_ * 5) for _ in (x, y, z)]

                pl = pv.Plotter(
                    off_screen=True,
                    border=False,
                    window_size=[768, 768],
                    line_smoothing=True,
                    point_smoothing=True,
                    polygon_smoothing=True,
                )
                pl.enable_parallel_projection()  # type: ignore
                pl.enable_depth_peeling()  # type: ignore
                pl.enable_anti_aliasing('msaa')  # type: ignore

                # 添加各种组件到场景
                if post_mesh_outlier is not None and len(post_mesh_outlier.faces):
                    pl.add_mesh(post_mesh_outlier, color='green')  # type: ignore
                pl.add_mesh(post_mesh, color='lightgreen')  # type: ignore

                pre_mesh: trimesh.Trimesh = bone_meshes[part][0].copy()
                pre_mesh.apply_transform(np.linalg.inv(matrix))  # 将术前网格逆变换到术后坐标系对比
                pl.add_mesh(pre_mesh, color='lightyellow')  # type: ignore
                pl.add_points(
                    vertices, color='crimson', render_points_as_spheres=True, point_size=3
                )  # 实际采样点（深红）

                pl.camera_position = 'xz'
                pl.reset_camera(bounds=b.T.flatten())  # type: ignore
                pl.camera.parallel_scale = (b[1][2] - b[0][2]) * 0.6
                pl.reset_camera_clipping_range()
                pl.render()

                # 渲染并拼接正侧位（AP & Lateral）视图
                sil = None
                if metal_meshes[part][1] is not None:
                    metal_actor = pl.add_mesh(metal_meshes[part][1], color='lightblue')  # type: ignore
                    sil = pl.add_silhouette(metal_actor.GetMapper().GetInput(), color='lightgray')  # type: ignore

                imgs = []
                # 循环生成两个视角的截图：正面 (0度) 和侧面 (90或-90度)
                for i, deg in enumerate([0, 90 if rl == 'R' else -90]):
                    for actor in pl.actors.values():
                        actor.SetVisibility(False)
                    if sil is not None:
                        sil.SetVisibility(True)

                    pl.window_size = [[wx, wy][i], h]

                    pl.camera_position = 'xz'
                    pl.reset_camera(bounds=b.T.flatten())  # type: ignore
                    pl.camera.Azimuth(deg)
                    pl.camera.parallel_scale = (b[1][2] - b[0][2]) * 0.6
                    pl.reset_camera_clipping_range()
                    pl.render()
                    a = np.array(pl.screenshot(return_img=True)).copy()

                    [pl.actors[_].SetVisibility(True) for _ in pl.actors].clear()

                    pl.reset_camera_clipping_range()
                    pl.render()
                    c = np.array(pl.screenshot(return_img=True)).copy()

                    # 将金属假体的剪影合并到截图上，确保假体始终可见
                    mask = (a != pl.background_color.int_rgb).any(axis=-1)
                    c[mask] = a[mask]
                    imgs.append(c)

                # 横向拼接图像并显示在 Streamlit
                st.image(np.hstack(imgs))

                pl.close()  # type: ignore

            if (_ := metal_meshes[part][1]) is None:
                st.warning('未发现金属假体')
            else:
                st.caption(f'金属体 {len(_.vertices)} 点 {len(_.faces)} 面')

        cols[0].space('medium')

    # 提交
    with cols[0]:
        with st.form('submit'):
            if pairs[prl].get('excluded', False):
                st.warning(f'已排除 {prl}')

            for k, v in save.items():
                if isinstance(v, dict) and k in saved and isinstance(saved[k], dict):
                    saved[k].update(v)
                else:
                    saved[k] = v
            st.code(tomlkit.dumps({'align': saved}), 'toml')

            if st.form_submit_button('提交（覆盖）' if save_key in saved else '提交'):
                # 更新内存中的总表
                pairs[prl]['align'] = saved

                f = root / 'pair' / pid / rl / 'align.toml'
                f.write_bytes(tomlkit.dumps(saved).encode('utf-8'))

                st.session_state.clear()
                st.session_state['init'] = cfg, pairs
                st.rerun()
