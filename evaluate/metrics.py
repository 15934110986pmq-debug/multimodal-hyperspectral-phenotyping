"""指标核验：模型精度与配准误差。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return float(r2), rmse


def registration_error(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> Dict[str, float]:
    """配准误差：RMS / 最大误差 / 均值误差（mm）。"""
    if source_xyz.shape != target_xyz.shape:
        raise ValueError("两组点云形状需一致")
    err = np.linalg.norm(np.asarray(source_xyz) - np.asarray(target_xyz), axis=1)
    return {
        "rms_mm": float(np.sqrt(np.mean(err**2))),
        "mean_mm": float(err.mean()),
        "max_mm": float(err.max()),
        "p95_mm": float(np.percentile(err, 95)),
    }


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    """把模型指标渲染成可打印/可写入报告的一段文本。"""
    lines: List[str] = []
    model_type = metrics.get("model_type", "?")
    folds = metrics.get("cv_folds", "?")
    lines.append(f"模型: {model_type} (CV {folds}-fold)")
    for name, r2, rmse in zip(
        metrics.get("trait_names", []),
        metrics.get("r2", []),
        metrics.get("rmse", []),
    ):
        lines.append(f"  {name:>16}: R^2 = {r2:.3f} | RMSE = {rmse:.3f}")
    avg_r2 = float(np.mean(metrics.get("r2", [0.0])))
    lines.append(f"  平均 R^2 = {avg_r2:.3f}")
    return "\n".join(lines)
