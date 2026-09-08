"""光谱特征提取：植被指数、红边位置、光谱特征向量。
+
输入输出都是 :class:`~data.schema.SampleSet`。输入 ``X`` 为逐样本反射率 (N, B)，
输出 ``X`` 替换为工程特征矩阵 (N, F)，并把 ``feature_names`` 放进 ``meta``。

支持 ``fit`` / ``transform`` 两阶段：在训练集上 ``fit`` 后，可把同一套特征
（尤其 PCA 基与中心）复用到新的场景，保证训练、预测的特征空间一致，从而支持
“用带标签数据训练、对无标签新场景反演”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from data.schema import SampleSet
from .base import Stage


def _nearest_band(wavelengths: np.ndarray, target: float) -> Tuple[int, float]:
    """返回与 target 最接近的波段索引及其波长。"""
    idx = int(np.argmin(np.abs(wavelengths - target)))
    return idx, float(wavelengths[idx])


@dataclass
class SpectrumFeatures(Stage):
    """从反射率光谱中提取一组工程特征，可 fit / transform 两阶段复用。"""

    name: str = "spectrum_features"
    n_components: int = 8                    # 保留的光谱主成分数
    band_ref: float = 660.0                  # NDVI 红波段参考
    band_nir: float = 800.0                  # NDVI 近红外参考
    feature_names: List[str] = field(default_factory=list)
    train_wavelengths: Optional[np.ndarray] = None   # 训练集波长（fit 时记录）
    _center: Optional[np.ndarray] = None             # 训练集光谱均值
    _pca_vt: Optional[np.ndarray] = None             # 训练集 PCA 基 (n_comp, B)
    _fitted: bool = False

    @staticmethod
    def _resample(X: np.ndarray, src_wl: np.ndarray, dst_wl: np.ndarray) -> np.ndarray:
        """把每条光谱从 src_wl 线性插值到 dst_wl（按波长，向量化）。"""
        X = np.asarray(X, dtype=float)
        src = np.asarray(src_wl, dtype=float)
        dst = np.asarray(dst_wl, dtype=float)
        if src.size == dst.size and np.allclose(src, dst, atol=1e-6):
            return X
        idx = np.searchsorted(src, dst)
        idx = np.clip(idx, 1, src.size - 1)
        x0 = src[idx - 1]
        x1 = src[idx]
        y0 = X[:, idx - 1]
        y1 = X[:, idx]
        w = np.where(x1 > x0, (dst - x0) / (x1 - x0), 0.0)
        return y0 + (y1 - y0) * w

    def _pca_n(self, n_bands: int) -> int:
        """实际保留的主成分数（受波段数约束）。"""
        if self.n_components <= 0:
            return n_bands
        return min(self.n_components, n_bands)

    def _index_names(self) -> List[str]:
        return ["NDVI", "NDRE", "PRI", "NDWI", "SR", "GNDVI"]

    def fit(self, X: np.ndarray, wavelengths: np.ndarray) -> "SpectrumFeatures":
        """在训练集上记录波长网格、光谱均值与 PCA 基，供后续预测复用。"""
        X = np.asarray(X, dtype=float)
        wl = np.asarray(wavelengths, dtype=float)
        self.train_wavelengths = wl.copy()
        self._center = X.mean(axis=0)
        n_comp = self._pca_n(X.shape[1])
        X_centered = X - self._center
        if n_comp < X.shape[1]:
            _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
            self._pca_vt = np.asarray(vt[:n_comp], dtype=float)
        else:
            self._pca_vt = np.eye(n_comp, dtype=float)[:n_comp]
        self.feature_names = (
            self._index_names()
            + ["RedEdgePos", "MeanRefl", "StdRefl"]
            + [f"PC{i+1}" for i in range(n_comp)]
        )
        self._fitted = True
        return self

    def _index_features(
        self,
        X: np.ndarray,
        wavelengths: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """计算一组比值/归一化植被指数。"""
        idx_red, _ = _nearest_band(wavelengths, self.band_ref)
        idx_nir, _ = _nearest_band(wavelengths, self.band_nir)

        i_531, _ = _nearest_band(wavelengths, 531.0)
        i_570, _ = _nearest_band(wavelengths, 570.0)   # PRI
        i_green, _ = _nearest_band(wavelengths, 550.0)
        i_1240, _ = _nearest_band(wavelengths, 1240.0) # 水分指数近红外2

        red = X[:, idx_red]
        nir = X[:, idx_nir]
        green = X[:, i_green]
        nir2 = X[:, i_1240]

        eps = 1e-8
        ndvi = (nir - red) / (nir + red + eps)
        # 归一化差值红边 (NDRE): (NIR - red_edge) / (NIR + red_edge)
        i_rededge, _ = _nearest_band(wavelengths, 705.0)
        re = X[:, i_rededge]
        ndre = (nir - re) / (nir + re + eps)
        pri = (X[:, i_531] - X[:, i_570]) / (X[:, i_531] + X[:, i_570] + eps)
        ndwi = (nir - nir2) / (nir + nir2 + eps)
        simple_ratio = nir / (red + eps)
        gndvi = (nir - green) / (nir + green + eps)

        feats = np.column_stack([ndvi, ndre, pri, ndwi, simple_ratio, gndvi])
        names = ["NDVI", "NDRE", "PRI", "NDWI", "SR", "GNDVI"]
        return feats, names

    def _red_edge_position(self, X: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """红边位置：在 680–780nm 内取一阶导数最大值对应波长。"""
        lo, _ = _nearest_band(wavelengths, 680.0)
        hi, _ = _nearest_band(wavelengths, 780.0)
        lo, hi = min(lo, hi), max(lo, hi)
        if hi - lo < 2:
            return np.zeros(X.shape[0])
        seg = X[:, lo : hi + 1]
        wl = wavelengths[lo : hi + 1]
        # 一阶导数（中心差分近似）
        deriv = np.gradient(seg, wl, axis=1)
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            out[i] = wl[int(np.argmax(deriv[i]))]
        return out

    def transform(self, X: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """用训练时记录的网格/均值/PCA 基对新光谱提特征。

        若 ``wavelengths`` 与训练网格不一致，会先把光谱重采样到训练网格再提特征，
        保证与训练特征同维、同基（含 PCA），使模型可跨场景应用。
        """
        X = np.asarray(X, dtype=float)
        wl = np.asarray(wavelengths, dtype=float)

        if self._fitted:
            Xa = self._resample(X, wl, self.train_wavelengths) if self.train_wavelengths is not None else X
            eff_wl = self.train_wavelengths if self.train_wavelengths is not None else wl
        else:
            Xa = X
            eff_wl = wl

        indices, _ = self._index_features(Xa, eff_wl)
        rep = self._red_edge_position(Xa, eff_wl).reshape(-1, 1)
        mean_frac = Xa.mean(axis=1, keepdims=True)
        std_frac = Xa.std(axis=1, keepdims=True)

        if self._fitted:
            X_centered = Xa - self._center
            pca = X_centered @ self._pca_vt.T
        else:
            n_comp = self._pca_n(Xa.shape[1])
            X_centered = Xa - Xa.mean(axis=0, keepdims=True)
            if n_comp < Xa.shape[1]:
                _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
                pca = X_centered @ vt[:n_comp].T
            else:
                pca = X_centered

        return np.column_stack([indices, rep, mean_frac, std_frac, pca])

    def run(self, data: SampleSet) -> SampleSet:
        data.validate()
        X = data.X.astype(float)
        wl = np.asarray(data.wavelengths, dtype=float)
        if not self._fitted:
            self.fit(X, wl)
        out_X = self.transform(X, wl)
        return SampleSet(
            X=out_X,
            wavelengths=wl,
            y=data.y,
            trait_names=data.trait_names,
            sample_ids=data.sample_ids,
            meta={**data.meta, "feature_names": self.feature_names, "mode": "features"},
        )
