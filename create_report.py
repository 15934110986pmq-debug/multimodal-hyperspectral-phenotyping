"""生成与部署演示系统内容一致的静态报告（HTML，内嵌图片，可离线打开）。
+
用法：
  python create_report.py --source real --data_root D:\\multimode\\data --model ridge
  python create_report.py --source synthetic --model pls --out static_report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loaders import LOADERS  # noqa: E402
from pipeline import run_phenotyping  # noqa: E402
from demo.scene import make_synthetic_scene  # noqa: E402


def _fig_to_b64(fig: plt.Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(html: str, caption: str = "") -> str:
    return f'<div class="fig"><img src="data:image/png;base64,{html}"/><div class="cap">{caption}</div></div>'


def load_data(source: str, data_root: str, data_path: str, trait: str, n_samples: int):
    trait_list = [t.strip() for t in trait.split(",") if t.strip()]
    if source == "synthetic":
        return make_synthetic_scene(), LOADERS["synthetic"](n_samples=n_samples).load()
    if source == "real":
        rs = LOADERS["real"](root=data_root, trait_columns=trait_list,
                             max_pixels=300000, max_samples=8000).load()
        return rs.scene, rs.samples
    if source == "greenhyperspectra":
        rs = LOADERS["greenhyperspectra"](path=data_path, trait_columns=trait_list).load()
        return make_synthetic_scene(), rs
    raise ValueError(source)


def make_figures(scene, ss, metrics, y_pred) -> Dict[str, str]:
    figs: Dict[str, str] = {}

    # 光谱
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.linspace(0, ss.n_samples - 1, min(8, ss.n_samples)).astype(int)
    for i in idx:
        ax.plot(ss.wavelengths, ss.X[i], lw=0.9, alpha=0.85)
    ax.set_title("样本反射率光谱"); ax.set_xlabel("波长 (nm)"); ax.set_ylabel("反射率"); ax.grid(alpha=0.3)
    figs["spectra"] = _fig_to_b64(fig)

    # R2 条形
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(metrics["trait_names"], metrics["r2"], color="#4c72b0")
    ax.axhline(0.9, color="r", ls="--", label="目标 0.9")
    ax.set_title("表型反演 R^2（交叉验证）"); ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20); ax.legend()
    figs["r2"] = _fig_to_b64(fig)

    # 预测 vs 实测（前 4 个表型）
    pv = []
    for j, name in enumerate(metrics["trait_names"][:4]):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        y = ss.y[:, j]; pred = y_pred[:, j]
        ax.scatter(y, pred, s=10, alpha=0.6)
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--")
        ax.set_title(f"{name}\nR^2={metrics['r2'][j]:.3f}")
        ax.set_xlabel("实测"); ax.set_ylabel("预测")
        pv.append(_fig_to_b64(fig))
    figs["pred"] = pv

    # 场景 RGB + 点云高程
    cube = np.nan_to_num(scene.cube)
    wl = scene.wavelengths
    i_ = lambda t: int(np.argmin(np.abs(wl - t)))
    rgb = np.stack([cube[:, :, i_(650)], cube[:, :, i_(550)], cube[:, :, i_(450)]], axis=-1)
    rgb = np.clip(rgb / (rgb.max(axis=2, keepdims=True) + 1e-9), 0, 1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.imshow(rgb); ax.set_title("RGB 合成 (650/550/450)"); ax.axis("off")
    figs["rgb"] = _fig_to_b64(fig)

    if scene.pointcloud is not None and len(scene.pointcloud.xyz):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        sc = ax.scatter(scene.pointcloud.xyz[:, 0], scene.pointcloud.xyz[:, 1],
                        c=scene.pointcloud.xyz[:, 2], s=1.5, cmap="viridis")
        ax.set_title("三维点云（按高程着色）"); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(sc, ax=ax, label="高程")
        figs["pc"] = _fig_to_b64(fig)
    return figs


def render_html(metrics, ss, figs, source, meta, report) -> str:
    rows = "".join(
        f"<tr><td>{n}</td><td>{r2:.3f}</td><td>{rmse:.3f}</td></tr>"
        for n, r2, rmse in zip(metrics["trait_names"], metrics["r2"], metrics["rmse"])
    )
    avg = float(np.mean(metrics["r2"]))
    pred_imgs = "".join(_img(b, "预测 vs 实测") for b in figs["pred"])
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"/>
<title>多模态高光谱植物表型获取系统 · 静态报告</title>
<style>
 body{{font-family:'Microsoft YaHei',sans-serif;margin:0;background:#f5f7fa;color:#1f2933;}}
 .wrap{{max-width:1100px;margin:0 auto;padding:28px 24px;}}
 h1{{color:#16324f;border-bottom:3px solid #4c72b0;padding-bottom:10px;}}
 h2{{color:#16324f;margin-top:30px;}}
 .card{{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-top:16px;}}
 table{{border-collapse:collapse;width:100%;}} th,td{{border:1px solid #e2e6ea;padding:8px 10px;text-align:left;}}
 th{{background:#eef2f7;}} .fig{{display:inline-block;margin:8px;vertical-align:top;}}
 .fig img{{max-width:100%;border-radius:6px;}} .cap{{text-align:center;color:#5c6b7a;font-size:13px;}}
 .metrics{{display:flex;gap:14px;flex-wrap:wrap;}} .m{{flex:1;min-width:140px;background:#fff;border-radius:8px;padding:14px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06);}}
 .m b{{font-size:26px;color:#4c72b0;}}
 footer{{margin-top:40px;color:#8a97a6;font-size:12px;}}
</style></head><body><div class="wrap">
<h1>🌱 多模态高光谱植物表型获取系统</h1>
<p>三维结构 + 高光谱同步成像 · 高精度配准 · 反射率校正 · 表型智能解析</p>
<div class="card"><b>数据源：</b>{source}　<b>样本数：</b>{ss.n_samples}　<b>光谱波段：</b>{ss.n_bands}
　<b>生成时间：</b>{datetime.now().strftime("%Y-%m-%d %H:%M")}<br/><br/>元信息：<code>{meta}</code></div>

<h2>核心指标</h2>
<div class="metrics">
 <div class="m"><b>{avg:.3f}</b><br/>平均 R^2</div>
 <div class="m"><b>{metrics['cv_folds']}</b><br/>交叉验证折数</div>
 <div class="m"><b>{metrics['model_type']}</b><br/>模型</div>
 <div class="m"><b>{max(metrics['r2']):.3f}</b><br/>最高 R^2</div>
</div>

<h2>表型预测精度</h2>
<div class="card"><table><tr><th>表型</th><th>R^2</th><th>RMSE</th></tr>{rows}</table>
<pre>{report}</pre></div>

<div class="card">{_img(figs['spectra'], '样本反射率光谱')}{_img(figs['r2'], '表型 R^2（交叉验证）')}</div>
<div class="card">{pred_imgs}</div>
<div class="card">{_img(figs['rgb'], 'RGB 合成')}{_img(figs.get('pc',''), '三维点云 / 高程')}</div>

<footer>本报告由 create_report.py 生成，内容与部署演示系统一致；可离线打开。图片内嵌，无需外部资源。</footer>
</div></body></html>"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=os.environ.get("MULTIMODE_SOURCE", "synthetic"),
                   choices=["synthetic", "real", "greenhyperspectra"])
    p.add_argument("--data_root", default=os.environ.get("MULTIMODE_DATA_ROOT", ""))
    p.add_argument("--data_path", default="")
    p.add_argument("--trait", default=os.environ.get(
        "MULTIMODE_TRAITS", "chlorophyll,LAI,water_content,biomass,soluble_solids"))
    p.add_argument("--model", default=os.environ.get("MULTIMODE_MODEL", "ridge"),
                   choices=["ridge", "random_forest", "gradient_boosting", "pls"])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--out", default="static_report.html")
    args = p.parse_args(argv)

    scene, ss = load_data(args.source, args.data_root, args.data_path, args.trait, args.n_samples)
    ss.validate()
    res = run_phenotyping(ss, model_type=args.model, cv_folds=args.cv)
    metrics = res["metrics"]
    figs = make_figures(scene, ss, metrics, res["y_pred"])
    html_str = render_html(metrics, ss, figs, args.source,
                           _short_meta(ss.meta), res["report"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str, encoding="utf-8")
    print(f"静态报告已生成 -> {out.resolve()}")
    print(res["report"])
    return 0


def _short_meta(meta: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in meta.items() if not isinstance(v, (list, dict)))


if __name__ == "__main__":
    sys.exit(main())
