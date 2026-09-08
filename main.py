"""端到端演示入口。
+
示例：
  .venv\\Scripts\\python main.py --loader synthetic --model random_forest
  .venv\\Scripts\\python main.py --loader greenhyperspectra --data <path> --trait chlorophyll,LAI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loaders import LOADERS
from pipeline import run_phenotyping


def _parse_list(value: str) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="多模态高光谱植物表型流水线")
    parser.add_argument("--loader", default="synthetic", choices=list(LOADERS.keys()))
    parser.add_argument("--data", default=None, help="公开数据集路径（greenhyperspectra 时必填）")
    parser.add_argument("--model", default="random_forest",
                        choices=["ridge", "random_forest", "gradient_boosting", "pls"])
    parser.add_argument("--trait", default=None, help="逗号分隔的表型列名")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    loader_cls = LOADERS[args.loader]
    if args.loader == "synthetic":
        loader = loader_cls(n_samples=args.n_samples)
        sample_set = loader.load()
    elif args.loader == "real":
        loader = loader_cls(root=args.data, trait_columns=_parse_list(args.trait),
                            max_pixels=300000, max_samples=8000)
        source = loader.load()
        sample_set = source.samples
    else:
        if not args.data:
            print("greenhyperspectra 需要 --data <路径>")
            return 1
        loader = loader_cls(path=args.data, trait_columns=_parse_list(args.trait))
        sample_set = loader.load()

    sample_set.validate()
    print(f"[数据] source={sample_set.meta.get('source')} samples={sample_set.n_samples} "
          f"bands={sample_set.n_bands} traits={sample_set.trait_names}")

    result = run_phenotyping(
        sample_set,
        model_type=args.model,
        cv_folds=args.cv,
        n_features=args.n_features,
    )
    print("\n" + result["report"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"model": result["metrics"], "n_features": len(result["feature_names"])},
            fh,
            ensure_ascii=False,
            indent=2,
        )

    if not args.no_plots:
        from demo.plots import (
            plot_spectra_sample,
            plot_predicted_vs_measured,
            plot_r2_bar,
        )
        plot_spectra_sample(sample_set, outdir / "spectra.png")
        plot_predicted_vs_measured(sample_set.y, result["y_pred"],
                                   sample_set.trait_names, outdir)
        plot_r2_bar(result["metrics"]["trait_names"], result["metrics"]["r2"],
                    outdir / "r2_bar.png")
        print(f"\n[图表] 已输出到 {outdir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
