"""生成变量施肥建议报告（处方图 + HTML，离线可打开）。

用法示例：
  .venv\\Scripts\\python create_fertilizer_report.py --source real --data_root D:\\multimode\\data --model ridge
  .venv\\Scripts\\python create_fertilizer_report.py --source synthetic --model pls --crop wheat
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from demo.fertilizer_report import make_fertilizer_report  # noqa: E402
from demo.scene import make_synthetic_scene  # noqa: E402
from loaders import LOADERS  # noqa: E402
from pipeline import run_phenotyping  # noqa: E402
from stages.fertilizer import (  # noqa: E402
    FertilizationAdvisor,
    build_prescription,
    prescription_from_scene,
    scene_ndvi,
)


def _load_data(source: str, data_root: str, trait: str, n_samples: int):
    trait_list = [t.strip() for t in trait.split(",") if t.strip()]
    if source == "synthetic":
        return make_synthetic_scene(), LOADERS["synthetic"](n_samples=n_samples).load()
    if source == "real":
        rs = LOADERS["real"](root=data_root, trait_columns=trait_list,
                             max_pixels=300000, max_samples=8000).load()
        return rs.scene, rs.samples
    raise ValueError(f"施肥处方报告支持 synthetic / real，当前：{source}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="生成变量施肥建议报告")
    p.add_argument("--source", default=os.environ.get("MULTIMODE_SOURCE", "real"),
                   choices=["synthetic", "real"])
    p.add_argument("--data_root", default=os.environ.get("MULTIMODE_DATA_ROOT", ""))
    p.add_argument("--trait", default=os.environ.get(
        "MULTIMODE_TRAITS", "chlorophyll,LAI,water_content,biomass,soluble_solids"))
    p.add_argument("--model", default=os.environ.get("MULTIMODE_MODEL", "ridge"),
                   choices=["ridge", "random_forest", "gradient_boosting", "pls"])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--crop", default="field_crop")
    p.add_argument("--proxy", default="chlorophyll", help="用于氮状态的表型列")
    p.add_argument("--pixel-mm", type=int, default=50)
    p.add_argument("--n-max", type=float, default=200.0)
    p.add_argument("--n-min", type=float, default=0.0)
    p.add_argument("--zones", type=int, default=5)
    p.add_argument("--outpng", default="output/fertilizer/fertilizer_prescription.png")
    p.add_argument("--outhtml", default="output/fertilizer/fertilizer_report.html")
    args = p.parse_args(argv)

    scene, ss = _load_data(args.source, args.data_root, args.trait, args.n_samples)
    ss.validate()
    advisor = FertilizationAdvisor(
        crop=args.crop, n_min=args.n_min, n_max=args.n_max,
        pixel_size_m=args.pixel_mm / 1000.0, n_zones=args.zones,
    )

    bands_match = (len(scene.wavelengths) == ss.n_bands
                   and np.allclose(np.asarray(scene.wavelengths, float),
                                   np.asarray(ss.wavelengths, float), atol=0.6))
    if (not bands_match or ss.y is None
            or (ss.trait_names and args.proxy not in ss.trait_names)):
        reason = "场景与训练样本波段不一致" if not bands_match else f"无标签或缺少 {args.proxy}"
        print(f"[注意] {reason}，改用 NDVI 指数法。")
        ndvi = scene_ndvi(scene)
        presc = build_prescription(ndvi, ndvi, advisor)
    else:
        res = run_phenotyping(ss, model_type=args.model, cv_folds=args.cv)
        trait_index = ss.trait_names.index(args.proxy)
        presc = prescription_from_scene(scene, res["model"], res["feature_extractor"],
                                        advisor, trait_index)
    presc["pixel_size_m"] = advisor.pixel_size_m

    out_png = Path(args.outpng)
    out_html = Path(args.outhtml)
    out = make_fertilizer_report(
        source=args.source, crop=args.crop, prescription=presc,
        out_png=out_png, out_html=out_html,
    )
    print(out["recommendations"])
    print(f"\n施肥处方报告已生成 -> {out_png.resolve()}")
    print(f"                          {out_html.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
