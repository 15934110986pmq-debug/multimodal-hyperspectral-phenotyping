"""数据预处理：坏点校正、去噪、辐射/反射率校正。
+
面向 :class:`~data.schema.RawScene`（高光谱立方体）。默认给出可用的默认实现，
也可替换为与硬件标定表耦合的更强实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from scipy.signal import savgol_filter

from data.schema import RawScene, Calibration
from .base import Stage


@dataclass
class BadPixelCorrection(Stage):
    """替换 NaN/Inf，并把沿光谱轴的孤立离群值拉回邻域中值。"""

    name: str = "bad_pixel_correction"
    threshold: float = 4.0   # 相对光谱中值超过该倍数视为坏点

    def run(self, scene: RawScene) -> RawScene:
        scene.validate()
        cube = np.nan_to_num(scene.cube.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        # 沿光谱轴的中值（每像素一个标量），广播到各波段再判坏点
        med = np.median(cube, axis=2)
        diff = np.abs(cube - med[:, :, None])
        mask = diff > (med[:, :, None] * self.threshold + 1e-6)
        if mask.any():
            cube[mask] = np.broadcast_to(med[:, :, None], cube.shape)[mask]
        scene.cube = cube
        scene.meta["bad_pixel_corrected"] = int(mask.sum())
        return scene


@dataclass
class Denoise(Stage):
    """沿光谱轴的 Savitzky-Golay 平滑去噪。"""

    name: str = "denoise"
    window: int = 9
    polyorder: int = 3

    def run(self, scene: RawScene) -> RawScene:
        scene.validate()
        w = min(self.window if self.window % 2 == 1 else self.window + 1, scene.cube.shape[2])
        if w < self.polyorder + 2:
            return scene
        flat = scene.cube.reshape(-1, scene.cube.shape[2])
        flat = savgol_filter(flat, window_length=w, polyorder=self.polyorder, axis=1)
        scene.cube = flat.reshape(scene.cube.shape)
        return scene


@dataclass
class ReflectanceCorrection(Stage):
    """用白板/暗电流做反射率校正：R = (DN - dark) / (white - dark)。
+
    当缺少白板时退化为归一化（除以光谱均值），保证下游可用。
    """

    name: str = "reflectance_correction"

    def run(self, scene: RawScene) -> RawScene:
        scene.validate()
        cal: Calibration | None = scene.calibration
        cube = scene.cube.astype(float)
        if cal is not None and cal.white_reference is not None:
            dark = cal.dark_reference if cal.dark_reference is not None else np.zeros_like(cal.white_reference)
            denom = cal.white_reference - dark + 1e-8
            refl = (cube - dark) / denom[None, None, :]
        else:
            # 无白板时的兜底：按逐像素光谱均值归一（无绝对反射率意义）
            refl = cube / (np.mean(cube, axis=2, keepdims=True) + 1e-8)
        refl = np.clip(refl, 0.0, 1.0)
        scene.cube = refl
        scene.meta["reflectance_corrected"] = True
        return scene
