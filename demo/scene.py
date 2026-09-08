"""合成植物场景生成器（演示用）。
+
在没有真实硬件数据的阶段，用它生成一个 "像样" 的单株冠层场景：
一个高光谱立方体 + 一个三维点云 + 一套定标参考，用来演示
采集 / RGB 合成 / 点云可视化 / 高程分析 / 光谱曲线等软件界面功能。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from data.schema import Calibration, PointCloud, RawScene
from loaders.synthetic import make_spectrum


def _leaf_mask(shape, n_leaves=7, seed=3, spread=60):
    """生成一个近似叶片叠加的布尔掩膜与高度场。"""
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = w / 2.0, h / 2.0
    mask = np.zeros(shape, dtype=bool)
    height = np.zeros(shape, dtype=float)
    for _ in range(n_leaves):
        ang = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0, spread)
        lx = cx + r0 * np.cos(ang)
        ly = cy + r0 * np.sin(ang)
        major = rng.uniform(15, 32)
        minor = rng.uniform(7, 14)
        rot = ang + rng.uniform(-0.6, 0.6)
        dx = x - lx
        dy = y - ly
        # 把椭圆压成一个有朝向的叶片
        u = dx * np.cos(rot) + dy * np.sin(rot)
        v = -dx * np.sin(rot) + dy * np.cos(rot)
        ellipse = (u / major) ** 2 + (v / minor) ** 2
        in_leaf = ellipse < 1.0
        mask |= in_leaf
        hh = np.clip(1.0 - ellipse, 0, 1) * rng.uniform(30, 80)
        height = np.maximum(height, np.where(in_leaf, hh, 0))
    return mask, height


def make_synthetic_scene(
    width: int = 160,
    height: int = 128,
    n_bands: int = 61,
    seed: int = 3,
) -> RawScene:
    """生成一个带点云和定标的合成冠层场景。"""
    wl = np.linspace(400.0, 1000.0, n_bands)
    mask, height_field = _leaf_mask((height, width), seed=seed)
    cube = np.zeros((height, width, n_bands))

    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    crad = np.sqrt((xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2)
    crad_n = crad / (0.5 * np.hypot(width, height) + 1e-9)

    rng = np.random.default_rng(seed)
    for r in range(height):
        for c in range(width):
            if not mask[r, c]:
                # 背景土壤：低反射、近灰
                base = 0.05 + 0.05 * _smooth_noise(r, c, rng)
                cube[r, c, :] = np.clip(base + rng.normal(0, 0.004, n_bands), 0, 1)
                continue
            # 叶片单点性状随径向位置变化（中心叶绿素更高、更潮）
            g = np.clip(1.0 - crad_n[r, c], 0, 1)
            chloro = 25 + 30 * g + rng.normal(0, 3)
            lai = 2.0 + 3.0 * g + rng.normal(0, 0.3)
            water = 60 + 25 * g + rng.normal(0, 2)
            bio = 30 + 50 * g + rng.normal(0, 4)
            ssc = 8 + 8 * g + rng.normal(0, 1)
            spec = make_spectrum(wl, chloro, lai, water, ssc, bio)
            cube[r, c, :] = np.clip(spec + rng.normal(0, 0.006, n_bands), 0, 1)

    # 点云：从叶片表面采样 (x, y, z)，z 越高越接近观测者
    idx = np.argwhere(mask)
    sel = idx[np.random.default_rng(seed).choice(len(idx), size=min(4000, len(idx)), replace=False)]
    xyz = np.zeros((len(sel), 3))
    for i, (r, c) in enumerate(sel):
        xyz[i, 0] = c
        xyz[i, 1] = r
        xyz[i, 2] = height_field[r, c]
    # 颜色由立方体三分量给出
    i_r, i_g, i_b = 17, 5, 2  # 约650/550/450nm
    colors = np.stack(
        [cube[sel[:, 0], sel[:, 1], i_r],
         cube[sel[:, 0], sel[:, 1], i_g],
         cube[sel[:, 0], sel[:, 1], i_b]],
        axis=-1,
    )
    colors = np.clip(colors / (colors.max(axis=1, keepdims=True) + 1e-9), 0, 1)

    # 定标：白板/暗背景
    cal = Calibration(
        wavelengths=wl,
        white_reference=np.clip(
            make_spectrum(wl, 55, 4.5, 88, 8, 90) + rng.normal(0, 0.01, n_bands), 0, 1
        ),
        dark_reference=np.full(n_bands, 0.01),
    )

    pointcloud = PointCloud(xyz=xyz[:, :3], colors=colors, meta={"source": "synthetic_scene"})
    return RawScene(
        cube=cube,
        wavelengths=wl,
        pointcloud=pointcloud,
        calibration=cal,
        meta={"source": "synthetic_scene", "width": width, "height": height},
    )


def _smooth_noise(r: int, c: int, rng: np.random.Generator) -> float:
    return -0.5 + ((r * 13 + c * 7) % 11) / 11.0
