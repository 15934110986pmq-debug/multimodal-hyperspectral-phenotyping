"""施肥建议 / 变量施肥处方图（Variable-Rate Nitrogen Prescription）。

思路（面向工程实现、贴近真实农艺）：
  1. 用已训练好的表型反演模型，对场景内**每个像素**预测其叶绿素（`chlorophyll`，
     是作物氮素营养状态的最强光谱代理）以及 LAI / 生物量。
  2. 结合归一化作物氮营养指数（Nitrogen Nutrition Index, NNI）：
       * ``chl_norm``   = 叶绿素相对参考窗的归一化值（0=严重缺氮，1=供氮充足）
       * ``ndvi_norm``  = 冠层长势（密度）贡献，用于区分"叶色浓但冠层稀"的情况
       * ``status``     = w_chl * chl_norm + w_ndvi * ndvi_norm
  3. 施肥量（kg N/ha）= ``n_min`` + (``n_max`` - ``n_min``) * (1 - status)。
     营养越缺 → 施得越多；营养充足 → 少施/不施。
  4. 把连续施氮量量化为分区处方（``n_zones`` 个等级），并统计各分区面积、
     占比与总需氮量，供"处方图 + 施肥建议报告"输出。

默认参数面向**大田籽粒作物**（可配置）。P/K 需结合土壤速效养分，当前由
叶片/冠层光谱无法直接反演，报告里会明确给出"需补充土壤化验"的边界说明。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.schema import RawScene, SampleSet
from .features import SpectrumFeatures


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def _nearest_band(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=float) - target)))


@dataclass
class FertilizationAdvisor:
    """由光谱反演表型生成变量施氮建议。参数可随作物类型调整。"""

    crop: str = "field_crop"            # 作物类型（仅用于报告展示与预设）
    chl_low: float = 15.0               # 叶绿素严重缺乏阈值（对应 NNI=0）
    chl_high: float = 55.0              # 叶绿素充足阈值（对应 NNI=1）
    n_min: float = 0.0                  # 最低施氮量 kg N/ha
    n_max: float = 200.0                # 最高施氮量 kg N/ha
    chl_weight: float = 0.7             # 叶绿素（氮状态）权重
    ndvi_weight: float = 0.3            # 冠层长势（NDVI）权重
    pixel_size_m: float = 0.05          # 像素地面尺寸（m/pixel），用于算面积
    n_zones: int = 5                    # 处方等级数

    def nitrogen_status(
        self,
        chlorophyll: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """归一化作物氮营养指数 status ∈ [0,1]，越大越不缺氮。"""
        chl = np.asarray(chlorophyll, dtype=float)
        chl_norm = _clip01((chl - self.chl_low) / (self.chl_high - self.chl_low + 1e-9))
        if ndvi is None:
            return chl_norm
        ndvi_norm = _clip01(ndvi)
        return _clip01(self.chl_weight * chl_norm + self.ndvi_weight * ndvi_norm)

    def recommend(self, status: np.ndarray) -> np.ndarray:
        """由营养指数给出施氮量（kg N/ha）。缺氮 → 多施；充足 → 少施。"""
        s = _clip01(status)
        return self.n_min + (self.n_max - self.n_min) * (1.0 - s)

    def zone_labels(self, rates: np.ndarray) -> np.ndarray:
        """把连续施氮量量化为 n_zones 个处方等级（0..n_zones-1）。"""
        edges = np.linspace(self.n_min, self.n_max, self.n_zones + 1)
        rates = np.asarray(rates, dtype=float)
        zone = np.searchsorted(edges, rates, side="right") - 1
        return np.clip(zone, 0, self.n_zones - 1)


def scene_ndvi(scene: RawScene, red=660.0, nir=800.0) -> np.ndarray:
    """逐像素 NDVI。"""
    cube = np.nan_to_num(scene.cube, nan=0.0)
    i_red = _nearest_band(scene.wavelengths, red)
    i_nir = _nearest_band(scene.wavelengths, nir)
    r = cube[:, :, i_red]
    n = cube[:, :, i_nir]
    return (n - r) / (n + r + 1e-8)


def predict_trait_map(
    scene: RawScene,
    model: Any,
    feature_extractor: SpectrumFeatures,
    trait_index: int,
) -> np.ndarray:
    """对场景逐像素做表型反演，返回 (H, W) 的预测值图。

    用与训练一致的 ``feature_extractor`` 二次提取特征后由 ``model`` 预测，
    因此要求场景与训练数据波长/波段设置一致（真实数据路径天然满足）；
    若波段不同，``feature_extractor.transform`` 会先把光谱重采样到训练网格，
    再复用训练时的均值与 PCA 基，从而实现跨场景/跨波段反演。
    """
    cube = np.nan_to_num(scene.cube, nan=0.0)
    h, w, b = cube.shape
    pixels = SampleSet(
        X=cube.reshape(-1, b),
        wavelengths=np.asarray(scene.wavelengths, dtype=float),
    )
    feat_mat = feature_extractor.transform(pixels.X, pixels.wavelengths)
    pred = np.asarray(model.predict(feat_mat), dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred[:, trait_index].reshape(h, w)


def build_prescription(
    trait_map: np.ndarray,
    ndvi_map: np.ndarray,
    advisor: FertilizationAdvisor,
) -> Dict[str, Any]:
    """由预测表型图 + NDVI 图生成处方图数据与分区统计。"""
    h, w = trait_map.shape
    status = advisor.nitrogen_status(trait_map, ndvi_map)
    rate = advisor.recommend(status)
    zone = advisor.zone_labels(rate)

    area_per_px = advisor.pixel_size_m ** 2
    total_area = area_per_px * h * w

    zone_table: List[Dict[str, float]] = []
    for z in range(advisor.n_zones):
        mask = zone == z
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        area = cnt * area_per_px
        zr = float(rate[mask].mean())
        zone_table.append({
            "zone": z,
            "n_rate": round(zr, 1),
            "area_m2": round(area, 1),
            "area_pct": round(area / total_area * 100.0, 1),
            "n_kg": round(area * zr, 1),
        })

    total_n = float((rate * area_per_px).sum())
    return {
        "status": status,
        "rate": rate,
        "zone": zone,
        "zone_table": zone_table,
        "total_area_m2": total_area,
        "total_n_kg": total_n,
        "mean_rate": float(rate.mean()),
        "mean_status": float(status.mean()),
    }


def prescription_from_scene(
    scene: RawScene,
    model: Any,
    feature_extractor: SpectrumFeatures,
    advisor: FertilizationAdvisor,
    trait_index: int,
) -> Dict[str, Any]:
    """端到端：场景 -> 逐像素表型 -> 处方图。"""
    trait_map = predict_trait_map(scene, model, feature_extractor, trait_index)
    ndvi_map = scene_ndvi(scene)
    return build_prescription(trait_map, ndvi_map, advisor)
