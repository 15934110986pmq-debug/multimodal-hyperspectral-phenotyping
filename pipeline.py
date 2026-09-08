"""一键流水线：数据 -> 特征 -> 反演 -> 指标。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from data.schema import SampleSet
from evaluate.metrics import summarize_metrics
from stages.base import Pipeline
from stages.features import SpectrumFeatures
from stages.phenotyping import PhenotypingModel
from stages.reflectance_correction import LeafAngleCorrection


def build_regression_pipeline(
    model_type: str = "ridge",
    cv_folds: int = 5,
    n_components: int = 12,
    angle_mode: Optional[str] = None,
    n_features: int = 8,
) -> Pipeline:
    """组装特征提取 + 可选的倾角校正 + 表型反演 pipeline。"""
    stages = [SpectrumFeatures(n_components=n_features)]
    if angle_mode:
        stages.append(LeafAngleCorrection(mode=angle_mode))
    stages.append(
        PhenotypingModel(
            model_type=model_type,
            cv_folds=cv_folds,
            n_components=n_components,
        )
    )
    return Pipeline(stages)


def run_phenotyping(
    data: SampleSet,
    *,
    model_type: str = "ridge",
    cv_folds: int = 5,
    n_components: int = 12,
    angle_mode: Optional[str] = None,
    n_features: int = 8,
) -> Dict[str, Any]:
    """跑通完整链路，返回包含特征表、模型与指标的字典。"""
    data.validate()
    pipeline = build_regression_pipeline(
        model_type=model_type,
        cv_folds=cv_folds,
        n_components=n_components,
        angle_mode=angle_mode,
        n_features=n_features,
    )
    result: SampleSet = pipeline.run(data)
    features = result.meta.get("feature_names", [])
    metrics = result.meta["metrics"]
    model = result.meta["model"]
    feature_extractor = pipeline.stages[0]   # SpectrumFeatures，用于逐像素预测

    # 用训练好的模型对训练集做一次预测，供可视化预测 vs 实测
    y_pred = model.predict(result.X)
    return {
        "pipeline": pipeline,
        "feature_names": features,
        "metrics": metrics,
        "model": model,
        "feature_extractor": feature_extractor,
        "data": result,
        "y_pred": y_pred,
        "report": summarize_metrics(metrics),
    }
