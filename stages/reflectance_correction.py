"""反射率校正：叶片倾角 / BRDF 视角影响校正。
+
叶片倾角与视角会改变叶片-传感器间的光路几何，影响同一叶片在不同视角下的反射率。
这里给出两件事：
+  * :class:`LeafAngleCorrection`：可插拔的校正 stage。默认提供"逐样本归一化到
    参考视角"的近似方案，并把 ``model`` 接口留出来，等有真数据再用 BRDF 拟合。
+  * :func:`cosine_correction`：基于视角几何的余弦校正常用近似，可直接用于
    多角度公开数据验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from data.schema import SampleSet
from .base import Stage


def cosine_correction(reflectance: np.ndarray, view_zenith_deg: np.ndarray) -> np.ndarray:
    """把反射率除以入射/视角余弦，近似补偿倾角几何效应。"""
    cos = np.cos(np.deg2rad(np.asarray(view_zenith_deg, dtype=float)))
    cos = np.where(np.abs(cos) < 1e-6, 1e-6, cos)
    return reflectance / cos[:, None]


@dataclass
class LeafAngleCorrection(Stage):
    """把叶片反射率校正到统一参考视角。
+
    ``mode="sample_norm"`` 时对每个样本做光谱均值归一（无外参时的稳健兜底）；
    ``mode="cosine"`` 时使用 :func:`cosine_correction`（需给定视角，验证 BRDF 用）。
    """

    name: str = "leaf_angle_correction"
    mode: str = "sample_norm"
    view_zenith: Optional[np.ndarray] = None    # (N,)

    def run(self, data: SampleSet) -> SampleSet:
        X = data.X.astype(float)
        if self.mode == "cosine":
            if self.view_zenith is None or len(self.view_zenith) != X.shape[0]:
                raise ValueError("cosine 模式需提供每样本视角 view_zenith")
            X = cosine_correction(X, self.view_zenith)
        else:
            X = X / (np.mean(X, axis=1, keepdims=True) + 1e-8)
        return SampleSet(
            X=np.clip(X, 0.0, 1.0),
            wavelengths=data.wavelengths,
            y=data.y,
            trait_names=data.trait_names,
            sample_ids=data.sample_ids,
            meta={**data.meta, "angle_corrected": self.mode},
        )
