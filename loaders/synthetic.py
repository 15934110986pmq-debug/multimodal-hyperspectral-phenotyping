"""合成数据生成器（离线兜底，默认可用）。
+
生成"物理合理"的叶片反射率光谱，以及对应的叶绿素、叶面积指数、含水量、生物量、
可溶性固形物等表型标签。关系是**学习可得**但带噪声，用来验证整条工程流水线与
建模链路能跑通、能出合理的 R²。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from data.schema import SampleSet
from .base import DataLoader


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _gauss(wl: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wl - center) / width) ** 2)


def make_spectrum(
    wl: np.ndarray,
    chlorophyll: float,
    lai: float,
    water: float,
    ssc: float,
    biomass: float,
) -> np.ndarray:
    """由表型量生成一条反射率曲线。
+
    * ``chlorophyll``：增强叶绿素吸收（红区 672nm 附近）。
    * ``lai``：增强近红外台地（多次散射）。
    * ``water``：增强 970nm 附近的水分吸收。
    * ``ssc``：轻微移动红边位置（可溶性固形物相关）。
    * ``biomass``：提高近红外整体水平（冠层密度/散射更强）。
    """
    base = np.zeros_like(wl)
    base += 0.05 * _gauss(wl, 450, 45)                 # 蓝区
    base += 0.15 * _gauss(wl, 552, 42)                 # 绿峰
    base += 0.46 * _sigmoid((wl - 705 - ssc * 1.2) / 22)  # 红边 -> 近红外台地

    chl = np.clip((chlorophyll - 20) / 40.0, 0, 1)
    wat = np.clip((water - 55) / 35.0, 0, 1)
    lai_n = np.clip((lai - 1) / 5.0, 0, 1)
    ssc_n = np.clip((ssc - 5) / 15.0, 0, 1)
    bio_n = np.clip((biomass - 15) / 85.0, 0, 1)

    # 叶绿素降低红区反射
    chl_abs = _gauss(wl, 672, 30)
    refl = base * (1 - 0.55 * chl * chl_abs)
    # 水分降低 970nm 附近反射
    wat_abs = _gauss(wl, 970, 80)
    refl = refl * (1 - 0.45 * wat * wat_abs)
    # LAI 提升近红外台地
    refl = refl + 0.20 * lai_n * _sigmoid((wl - 705) / 20)
    # biomass 提升 750–1000nm 整体反射水平（与 LAI 波形略有差异）
    refl = refl + 0.16 * bio_n * _gauss(wl, 880, 140)
    # 可溶性固形物轻微压低整体（一个可学习的小偏移）
    refl = refl * (1 - 0.08 * ssc_n)

    return np.clip(refl, 0.0, 1.0)


class SyntheticLoader(DataLoader):
    """生成 :class:`SyntheticLoader` 风格的合成表型数据。"""

    name = "synthetic"

    def __init__(
        self,
        n_samples: int = 400,
        wl_start: float = 400.0,
        wl_end: float = 1000.0,
        n_bands: int = 121,
        noise: float = 0.012,
        random_state: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.wl_start = wl_start
        self.wl_end = wl_end
        self.n_bands = n_bands
        self.noise = noise
        self.random_state = random_state
        self.trait_names = ["chlorophyll", "LAI", "water_content", "biomass", "soluble_solids"]

    def load(self, **kwargs: Any) -> SampleSet:
        rng = np.random.default_rng(self.random_state)
        wl = np.linspace(self.wl_start, self.wl_end, self.n_bands)
        n = self.n_samples

        chlorophyll = rng.uniform(20, 60, n)
        lai = rng.uniform(1, 6, n)
        water = rng.uniform(55, 92, n)
        biomass = rng.uniform(15, 100, n)
        ssc = rng.uniform(5, 20, n)
        traits = np.column_stack([chlorophyll, lai, water, biomass, ssc])

        X = np.zeros((n, self.n_bands))
        for i in range(n):
            spec = make_spectrum(wl, chlorophyll[i], lai[i], water[i], ssc[i], biomass[i])
            spec = spec + rng.normal(0, self.noise, spec.size)
            X[i] = np.clip(spec, 0, 1)

        # 让 LAI 与近红外呈正相关、含水量与 970nm 负相关的信号更强，保证可学习性
        if self.n_bands > 5:
            i970 = int(np.argmin(np.abs(wl - 970)))
            X[:, i970] = X[:, i970] * (1 - 0.3 * (water - water.min()) / (np.ptp(water) + 1e-9))

        return SampleSet(
            X=X,
            wavelengths=wl,
            y=traits,
            trait_names=self.trait_names,
            sample_ids=[f"syn_{i:04d}" for i in range(n)],
            meta={"source": "synthetic", "noise": self.noise, "n_bands": self.n_bands},
        )
