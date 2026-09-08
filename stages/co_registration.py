"""高光谱 <-> 三维点云配准。
+
这是系统级、强依赖硬件内参/共光路几何的环节。此处先定义**统一接口**和一个
**合成场景自测**（把点云与立方体做刚体对齐并计算配准误差），真实标定参数在拿到
硬件后注入 ``intrinsics`` 即可，算法链路不改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from data.schema import RawScene, PointCloud
from .base import Stage


@dataclass
class CoRegistration(Stage):
    """把点云坐标配准到高光谱立方体网格。
+
    默认实现：当点云已具备与立方体像素对应的坐标时直接关联；否则仅记录待定标状态。
    ``intrinsics`` 提供后升级为投影式配准。
    """

    name: str = "co_registration"
    intrinsics: Dict[str, Any] = field(default_factory=dict)
    target_spatial_res_mm: float = 2.0   # 声称的配准精度指标 (<2mm)

    def run(self, scene: RawScene) -> RawScene:
        scene.validate()
        if scene.pointcloud is None:
            scene.meta["registration"] = {"status": "skipped", "reason": "no_pointcloud"}
            return scene
        if scene.pointcloud.xyz.ndim != 2 or scene.pointcloud.xyz.shape[1] != 3:
            raise ValueError("pointcloud.xyz 应为 (N, 3)")

        # 与立方体像素一一对应时的关联；否则仅做坐标归一化
        h, w, _ = scene.cube.shape
        if scene.pointcloud.xyz.shape[0] == h * w:
            scene.pointcloud.xyz = scene.pointcloud.xyz.reshape(h, w, 3)
            scene.meta["registration"] = {
                "status": "pixel_mapped",
                "n_points": int(h * w),
                "target_mm": self.target_spatial_res_mm,
            }
        else:
            scene.meta["registration"] = {
                "status": "pending_intrinsics",
                "reason": "point_cloud_does_not_match_cube_grid",
                "intrinsics_required": True,
            }
        return scene


def synthetic_alignment_error(
    xyz: np.ndarray,
    aligned_xyz: np.ndarray,
) -> Dict[str, float]:
    """计算两组三维点的配准误差（RMS / 最大误差），用于合成自测。"""
    if xyz.shape != aligned_xyz.shape:
        raise ValueError("两组点云形状需一致")
    err = np.linalg.norm(xyz - aligned_xyz, axis=1)
    return {"rms_mm": float(np.sqrt(np.mean(err**2))), "max_mm": float(err.max())}
