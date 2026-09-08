"""统一内部数据模型。
+
整个流水线只认这里定义的类型。硬件细节、不同数据集的字段差异都在 ``loaders``
这一层被翻译成统一格式，换数据不需要改动算法链路。
+
两种"世界"在此分隔：
+  * 采集/标定级（``RawScene`` / ``PointCloud`` / ``Calibration``）：描述一次成像场景，
    服务于预处理、配准、反射率校正这些面向"立方体/点云"的 stage。
+  * 样本级（``SpectralSample`` / ``SampleSet``）：把每个植株/区域抽成一行光谱 + 表型标签，
    服务于特征提取和表型反演。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Calibration:
    """辐射/反射率定标所需参考物和几何参数。"""

    wavelengths: np.ndarray                       # (B,)
    white_reference: Optional[np.ndarray] = None  # (B,) 标准白板光谱
    dark_reference: Optional[np.ndarray] = None   # (B,) 暗电流/暗背景
    sensor_geometry: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.white_reference is not None and self.white_reference.shape[0] != self.wavelengths.shape[0]:
            raise ValueError("white_reference 波段数与 wavelengths 不一致")
        if self.dark_reference is not None and self.dark_reference.shape[0] != self.wavelengths.shape[0]:
            raise ValueError("dark_reference 波段数与 wavelengths 不一致")


@dataclass
class PointCloud:
    """单帧 / 单场景三维点云。"""

    xyz: np.ndarray                                # (N, 3) 点坐标
    intensity: Optional[np.ndarray] = None         # (N,) 强度
    reflectance: Optional[np.ndarray] = None       # (N, B) 每点挂接光谱（配准后可为空）
    colors: Optional[np.ndarray] = None            # (N, 3) RGB
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawScene:
    """一次成像的原始场景：高光谱立方体 + 可选点云 + 定标信息。"""

    cube: np.ndarray                                # (H, W, B) 辐射亮度/未校正 DN
    wavelengths: np.ndarray                         # (B,)
    pointcloud: Optional[PointCloud] = None
    calibration: Optional[Calibration] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.cube.ndim != 3:
            raise ValueError("cube 应为 (H, W, B) 三维数组")
        if self.cube.shape[2] != self.wavelengths.shape[0]:
            raise ValueError("cube 波段数与 wavelengths 不一致")


@dataclass
class SpectralSample:
    """单个植株/区域的反射率光谱 + 可选实测表型。"""

    wavelengths: np.ndarray                         # (B,)
    reflectance: np.ndarray                         # (B,) 校正后反射率
    traits: Optional[np.ndarray] = None             # (T,) 实测表型
    trait_names: Optional[List[str]] = None
    sample_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleSet:
    """一批样本的紧凑矩阵表示，是特征/模型 stage 的输入输出主体。"""

    X: np.ndarray                                    # (N, B) 反射率（或特征）
    wavelengths: np.ndarray                          # (B,)
    y: Optional[np.ndarray] = None                   # (N, T) 表型标签
    trait_names: Optional[List[str]] = None
    sample_ids: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_bands(self) -> int:
        return self.X.shape[1]

    def validate(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("X 应为 (N, B) 二维数组")
        if self.X.shape[1] != self.wavelengths.shape[0]:
            raise ValueError("X 的波段数与 wavelengths 不一致")
        if self.y is not None:
            if self.y.shape[0] != self.X.shape[0]:
                raise ValueError("X 与 y 的样本数不一致")
            if self.trait_names is not None and self.y.shape[1] != len(self.trait_names):
                raise ValueError("y 的列数与 trait_names 不一致")

    def subset(self, indices: np.ndarray) -> "SampleSet":
        return SampleSet(
            X=self.X[indices],
            wavelengths=self.wavelengths,
            y=None if self.y is None else self.y[indices],
            trait_names=self.trait_names,
            sample_ids=None if self.sample_ids is None else [self.sample_ids[i] for i in indices],
            meta=self.meta,
        )
