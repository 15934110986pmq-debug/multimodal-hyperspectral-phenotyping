"""生成"真实格式"演示数据集（内容为占位，结构同真实硬件输出）。
+
写入：
  data/hyperspectral_envi.hdr + .img   （ENVI BSQ float32 高光谱立方体）
  data/points.csv                       （点云 x,y,z,reflectance）
  data/labels.csv                       （逐像素 x,y + 表型实测）
+
用于在没有真实数据时可先验证整套"真实数据摄取 -> 建模"链路；真实数据接入时
只需用同一目录/格式替换即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loaders.synthetic import make_spectrum  # noqa: E402


def main(width: int = 160, height: int = 128, bands: int = 61, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    wl = np.linspace(400.0, 1000.0, bands)
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    r0 = np.sqrt((xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2)
    g = np.clip(1.0 - r0 / (0.5 * np.hypot(width, height)), 0, 1)

    chlorophyll = 25 + 30 * g + rng.normal(0, 3, (height, width))
    lai = 2.0 + 3.0 * g + rng.normal(0, 0.3, (height, width))
    water = 60 + 25 * g + rng.normal(0, 2, (height, width))
    biomass = 30 + 50 * g + rng.normal(0, 4, (height, width))
    ssc = 8 + 8 * g + rng.normal(0, 1, (height, width))

    cube = np.zeros((height, width, bands), dtype=np.float32)
    for r in range(height):
        for c in range(width):
            spec = make_spectrum(wl, chlorophyll[r, c], lai[r, c], water[r, c],
                                 ssc[r, c], biomass[r, c])
            cube[r, c, :] = np.clip(spec + rng.normal(0, 0.006, bands), 0, 1)

    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)

    # 写 ENVI（BSQ, float32）
    hdr = out / "hyperspectral_envi.hdr"
    dat = out / "hyperspectral_envi.img"
    cube.tofile(dat)
    wl_str = ", ".join(f"{x:.2f}" for x in wl)
    hdr.write_text(
        "ENVI\n"
        f"description = {{ Synthetic_placeholder_cube }}\n"
        f"samples = {width}\n"
        f"lines = {height}\n"
        f"bands = {bands}\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 4\n"
        "interleave = bip\n"
        "byte order = 0\n"
        "wavelength units = Nanometers\n"
        f"wavelength = {{{wl_str}}}\n",
        encoding="utf-8",
    )

    # 点云（取叶片相关像素子集）
    mask = g > 0.05
    idx = np.argwhere(mask)
    sel = idx[rng.choice(len(idx), size=min(3000, len(idx)), replace=False)]
    z = np.clip(g[sel[:, 0], sel[:, 1]], 0, 1) * 120
    pc = pd.DataFrame({
        "x": sel[:, 1].astype(float),
        "y": sel[:, 0].astype(float),
        "z": z,
        "reflectance": cube[sel[:, 0], sel[:, 1], 50],
    })
    pc.to_csv(out / "points.csv", index=False)

    # 逐像素表型标签（行序 = 像素行主序，与立方体展平一致）
    labels = pd.DataFrame({
        "pixel": np.arange(height * width),
        "x": np.tile(np.arange(width), height),
        "y": np.repeat(np.arange(height), width),
        "chlorophyll": chlorophyll.ravel(),
        "LAI": lai.ravel(),
        "water_content": water.ravel(),
        "biomass": biomass.ravel(),
        "soluble_solids": ssc.ravel(),
    })
    labels.to_csv(out / "labels.csv", index=False)
    print(f"已生成真实格式数据集 -> {out}")
    print(f"  立方体 {cube.shape}，点云 {len(pc)}，标签 {len(labels)}")


if __name__ == "__main__":
    main()
