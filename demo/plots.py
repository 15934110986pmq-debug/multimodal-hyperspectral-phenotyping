"""可视化：光谱曲线、预测 vs 实测、R² 条形图、伪彩图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

from data.schema import SampleSet  # noqa: E402


def plot_spectra_sample(sample: SampleSet, out_path: Path, n: int = 8) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, sample.n_samples - 1, min(n, sample.n_samples)).astype(int)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in idx:
        ax.plot(sample.wavelengths, sample.X[i], lw=1.0, alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Sample reflectance spectra")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_predicted_vs_measured(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_names: List[str],
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for j, name in enumerate(trait_names):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(y_true[:, j], y_pred[:, j], s=12, alpha=0.6)
        lo = min(y_true[:, j].min(), y_pred[:, j].min())
        hi = max(y_true[:, j].max(), y_pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ss_res = np.sum((y_true[:, j] - y_pred[:, j]) ** 2)
        ss_tot = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean((y_true[:, j] - y_pred[:, j]) ** 2)))
        ax.set_title(f"{name}\nR^2={r2:.3f}  RMSE={rmse:.3f}")
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"pred_{j}_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)
    return files


def plot_r2_bar(trait_names: List[str], r2: List[float], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(trait_names, r2, color="#4c72b0")
    ax.axhline(0.9, color="r", ls="--", lw=1, label="target 0.9")
    ax.set_ylabel("R^2")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-validated phenotype R^2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def save_false_color(rgb: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    rgb = np.clip(rgb, 0, 1)
    ax.imshow(rgb)
    ax.set_title("False-color composite")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_prescription_map(prescription: dict, out_path: Path, pixel_size_m: float = 0.05) -> Path:
    """处方图：左为氮营养指数，右为变量施氮量/分区。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = prescription["status"]
    rate = prescription["rate"]
    zone = prescription["zone"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    im0 = axes[0].imshow(status, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_title("氮营养指数 (NNI)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rate, cmap="viridis")
    axes[1].set_title("施氮量 (kg N/ha)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(zone, cmap="tab10", vmin=0, vmax=max(1, int(zone.max())))
    axes[2].set_title("施肥分区")

    for ax in axes:
        ax.set_xlabel("columns"); ax.set_ylabel("rows")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"变量施肥处方图 (像素精度 {pixel_size_m*1000:.0f} mm)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
