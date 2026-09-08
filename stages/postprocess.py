"""表型输出：伪彩图、三维着色、报表汇总。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from data.schema import RawScene, SampleSet
from .base import Stage


@dataclass
class FalseColorComposite(Stage):
    """从高光谱立方体生成伪彩 RGB（默认 R=650, G=550, B=450）。"""

    name: str = "false_color"
    r_wl: float = 650.0
    g_wl: float = 550.0
    b_wl: float = 450.0

    def run(self, scene: RawScene) -> RawScene:
        scene.validate()
        wl = scene.wavelengths

        def nearest(t: float) -> int:
            return int(np.argmin(np.abs(wl - t)))

        rgb = np.stack(
            [
                scene.cube[:, :, nearest(self.r_wl)],
                scene.cube[:, :, nearest(self.g_wl)],
                scene.cube[:, :, nearest(self.b_wl)],
            ],
            axis=-1,
        )
        scene.meta["false_color"] = rgb
        return scene


@dataclass
class PhenotypeSummary(Stage):
    """把表型结果整理成可输出的摘要。"""

    name: str = "phenotype_summary"

    def run(self, data: SampleSet) -> Dict[str, Any]:
        data.validate()
        traits = data.trait_names or [f"trait_{j}" for j in range(data.y.shape[1])]
        out = {"n_samples": data.n_samples, "traits": {}}
        for j, name in enumerate(traits):
            col = data.y[:, j] if data.y is not None else None
            pred = None
            out["traits"][name] = {
                "mean_label": float(col.mean()) if col is not None else None,
                "std_label": float(col.std()) if col is not None else None,
            }
        return out
