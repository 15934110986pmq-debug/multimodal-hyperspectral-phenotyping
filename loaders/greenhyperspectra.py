"""公开数据集 GreenHyperSpectra 适配器。
+
数据集特征：多源高光谱 -> 全球植被表型预测，光谱约 400–2450 nm、1700+ 波段，
带反射率与植被表型标签（Hugging Face 仓库 waveyellow/GreenHyperSpectra）。
+
本适配器读取一个本地 CSV/Parquet 文件：光谱列以波长数值命名（如 ``400``, ``401``...），
表型列在 ``trait_columns`` 里指定。联网下载真实数据后，把文件路径传给 ``--data`` 即可。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from data.schema import SampleSet
from .base import DataLoader


class GreenHyperSpectraLoader(DataLoader):
    """读取 GreenHyperSpectra 风格的宽表（波长列为数值列名）。
+
    若环境缺少 ``pandas`` 会自动提示安装。数据列按列名是否可解析为 380–2500 之间的
    浮点数来识别为光谱波段。
    """

    name = "greenhyperspectra"

    def __init__(
        self,
        path: str,
        trait_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None,
        spectral_min: float = 380.0,
        spectral_max: float = 2500.0,
    ) -> None:
        self.path = path
        self.trait_columns = trait_columns or []
        self.id_column = id_column
        self.spectral_min = spectral_min
        self.spectral_max = spectral_max

    def _resolve_columns(self, columns: List[str]) -> Dict[str, Any]:
        spec_cols: List[str] = []
        for c in columns:
            try:
                wln = float(c)
            except ValueError:
                continue
            if self.spectral_min <= wln <= self.spectral_max:
                spec_cols.append(c)
        if not spec_cols:
            raise ValueError("未能从表头识别出光谱波段列（数值列名需落在光谱范围内）")
        spec_cols.sort(key=lambda c: float(c))
        traits = [c for c in self.trait_columns if c in columns]
        if not traits:
            candidate = [c for c in columns if c not in spec_cols and (self.id_column is None or c != self.id_column)]
            traits = candidate[:3]
        return {"spec_cols": spec_cols, "traits": traits}

    def load(self, **kwargs: Any) -> SampleSet:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 pandas，请安装：pip install pandas") from exc

        if self.path.endswith(".parquet"):
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path)

        cols = self._resolve_columns(list(df.columns))
        spec_cols, trait_cols = cols["spec_cols"], cols["traits"]
        wavelengths = np.array([float(c) for c in spec_cols])
        X = df[spec_cols].to_numpy(dtype=float)

        y = None
        trait_names = None
        if trait_cols:
            y = df[trait_cols].to_numpy(dtype=float)
            trait_names = trait_cols

        sample_ids = None
        if self.id_column and self.id_column in df.columns:
            sample_ids = df[self.id_column].astype(str).tolist()

        return SampleSet(
            X=X,
            wavelengths=wavelengths,
            y=y,
            trait_names=trait_names,
            sample_ids=sample_ids,
            meta={"source": self.name, "path": self.path, "n_bands": len(spec_cols)},
        )
