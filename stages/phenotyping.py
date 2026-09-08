"""表型反演模型（ML/DL 占位）。
+
输入 :class:`~data.schema.SampleSet`（X 为特征矩阵），输出带 ``model`` 与
交叉验证指标的对象。支持岭回归 / 随机森林 / 梯度提升 / PLS 等常见回归器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.schema import SampleSet
from .base import Stage


@dataclass
class PhenotypingModel(Stage):
    """封装训练好的表型预测器与评估结果。"""

    name: str = "phenotyping_model"
    model_type: str = "ridge"                # ridge | random_forest | gradient_boosting | pls
    random_state: int = 42
    cv_folds: int = 5
    n_components: int = 12
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    def _make_model(self):
        if self.model_type == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, random_state=self.random_state
            )
        if self.model_type == "pls":
            return make_pipeline(StandardScaler(), PLSRegression(n_components=self.n_components))
        raise ValueError(f"未知模型类型: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhenotypingModel":
        self.model = self._make_model()
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] == 1:
            self.model.fit(X, y.ravel())
        else:
            # 多输出：逐列训练，便于报告每个表型
            self.model = {"multi": True, "models": [self._make_model() for _ in range(y.shape[1])]}
            for j, m in enumerate(self.model["models"]):
                m.fit(X, y[:, j])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.model, dict):
            return np.column_stack([m.predict(X) for m in self.model["models"]])
        pred = self.model.predict(X)
        return np.asarray(pred).reshape(-1, 1)

    def run(self, data: SampleSet) -> SampleSet:
        # 特征提取后 X 的列已是特征而非波段，不必再校验与 wavelengths 等长
        if data.X.ndim != 2:
            raise ValueError("X 应为 (N, F) 二维特征矩阵")
        if data.y is not None and data.y.shape[0] != data.X.shape[0]:
            raise ValueError("X 与 y 的样本数不一致")
        if data.y is None:
            raise ValueError("表型反演需要带标签的 SampleSet")

        X, y = data.X.astype(float), data.y.astype(float)
        n_traits = y.shape[1]
        trait_names = data.trait_names or [f"trait_{j}" for j in range(n_traits)]

        self.feature_names = list(data.meta.get("feature_names", [f"f{i}" for i in range(X.shape[1])]))

        # 交叉验证评估（逐表型），同时训练最终模型
        from sklearn.model_selection import KFold
        r2_list: List[float] = []
        rmse_list: List[float] = []
        kf = KFold(n_splits=min(self.cv_folds, max(2, data.n_samples)), shuffle=True, random_state=self.random_state)
        for t in range(n_traits):
            yt = y[:, t]
            oof = np.empty_like(yt)
            for tr, te in kf.split(X):
                m = self._make_model()
                m.fit(X[tr], yt[tr])
                oof[te] = m.predict(X[te])
            r2 = 1.0 - np.sum((yt - oof) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)
            rmse = float(np.sqrt(np.mean((yt - oof) ** 2)))
            r2_list.append(float(r2))
            rmse_list.append(rmse)

        # 训练最终模型
        self.fit(X, y)
        self.metrics = {
            "model_type": self.model_type,
            "trait_names": trait_names,
            "r2": r2_list,
            "rmse": rmse_list,
            "cv_folds": self.cv_folds,
        }
        return SampleSet(
            X=X,
            wavelengths=data.wavelengths,
            y=data.y,
            trait_names=data.trait_names,
            sample_ids=data.sample_ids,
            meta={**data.meta, "mode": "phenotyping", "model": self, "metrics": self.metrics},
        )

    def report_metrics(self) -> str:
        out = [f"模型: {self.model_type} (CV {self.cv_folds}-fold)"]
        for name, r2, rmse in zip(self.metrics["trait_names"], self.metrics["r2"], self.metrics["rmse"]):
            out.append(f"  {name:>12}: R^2 = {r2:.3f} | RMSE = {rmse:.3f}")
        return "\n".join(out)
