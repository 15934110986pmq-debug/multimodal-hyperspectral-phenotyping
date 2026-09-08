"""真实数据摄取器：读取硬件/处理产出的原始格式。
+
支持：
  * 高光谱立方体：ENVI (``.hdr`` + ``.img``/``.dat``) 或 GeoTIFF (``.tif``) 或 ``.npy``
  * 点云：``.csv`` / ``.npy``（列 ``x,y,z`` 及可选 ``reflectance``/``r,g,b``）
  * 表型标签：``.csv``（含样本 id 列与若干表型列）
+
统一输出 :class:`~data.schema.RawScene`（可视化/预处理）与
:class:`~data.schema.SampleSet`（特征/建模）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from data.schema import Calibration, PointCloud, RawScene, SampleSet
from .base import DataLoader


@dataclass
class RealSource:
    scene: RawScene
    samples: SampleSet


def _read_envi(hdr_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """解析 ENVI .hdr，读取与头同名的数据文件。支持 BSQ / BIL / BIP。

    数据文件扩展名常见有 ``.img`` / ``.dat`` / ``.bil`` / ``.bsq`` / ``.bip``。
    不同硬件厂商命名略有差异，这里按".hdr 同 stem"去匹配，提高兼容性。
    """
    meta: Dict[str, str] = {}
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip().lower()] = v.strip()

    samples = int(meta.get("samples", "0"))
    lines = int(meta.get("lines", "0"))
    bands = int(meta.get("bands", "0"))
    if not (samples and lines and bands):
        raise ValueError("ENVI 头缺失 samples/lines/bands")
    interleave = meta.get("interleave", "bsq").upper()
    dtype_name = meta.get("data type", "4")
    dtype_map = {"1": np.uint8, "2": np.int16, "3": np.int32, "4": np.float32,
                 "5": np.float64, "12": np.uint16, "13": np.uint32}
    dtype = dtype_map.get(dtype_name, np.float32)

    img = None
    # 优先按"头文件去掉 .hdr"得到的同名数据文件（例如 guanmu.bil.hdr -> guanmu.bil）
    base = hdr_path.with_suffix("")
    if base.exists():
        img = base
    else:
        for ext in (".bil", ".img", ".dat", ".bsq", ".bip"):
            cand = hdr_path.with_suffix(ext)
            if cand.exists():
                img = cand
                break
    if img is None:
        raise FileNotFoundError(
            f"找不到与 {hdr_path} 对应的影像文件（尝试了 .bil/.img/.dat/.bsq/.bip）"
        )
    file_size = img.stat().st_size
    expected = int(samples) * int(lines) * int(bands) * np.dtype(dtype).itemsize
    # 大文件用 memmap 懒加载，避免整块读入内存（如几万行的高光谱扫描条带）。
    if file_size > 256 * 1024 * 1024:
        raw = np.memmap(img, dtype=dtype, mode="r")
    else:
        raw = np.fromfile(img, dtype=dtype)

    if interleave == "BSQ":
        cube = raw.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "BIL":
        cube = raw.reshape(lines, bands, samples).transpose(0, 2, 1)
    elif interleave == "BIP":
        cube = raw.reshape(lines, samples, bands)
    else:
        raise ValueError(f"不支持的 Interleave: {interleave}")

    if expected != file_size:
        # 并不强校验；仅提示已知的尺寸不一致，避免后续静默错位。
        pass

    wl_str = meta.get("wavelength", "").strip("{}")
    if wl_str:
        wavelengths = np.array([float(x) for x in wl_str.split(",") if x.strip()])
    else:
        wl0 = float(meta.get("wavelength units", "nan"))
        wavelengths = np.arange(bands, dtype=float)  # 未知波长时用索引作占位
    meta_out = {
        "interleave": interleave,
        "data_type": dtype_name,
        "dtype": dtype,
        "ceiling": meta.get("ceiling"),
        "bit_depth": meta.get("bit depth"),
        "wavelength_units": meta.get("wavelength units", ""),
        "label": meta.get("label", ""),
    }
    return cube, wavelengths, meta_out


def _read_cube(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if path.is_dir():
        hdr = next(path.glob("*.hdr"), None)
        bil = next(path.glob("*.bil"), None) or next(path.glob("*.bsq"), None) \
            or next(path.glob("*.bip"), None)
        tif = next(path.glob("*.tif"), None) or next(path.glob("*.tiff"), None)
        npy = next(path.glob("*.npy"), None)
    else:
        hdr = path if path.suffix.lower() == ".hdr" else None
        bil = path if path.suffix.lower() in (".bil", ".bsq", ".bip") else None
        tif = path if path.suffix.lower() in (".tif", ".tiff") else None
        npy = path if path.suffix.lower() == ".npy" else None

    if hdr is not None:
        return _read_envi(hdr)
    if bil is not None:
        # 没有 .hdr 的裸 .bil/.bsq/.bip：按 BIL 解析（波长未知用索引占位）
        arr = np.fromfile(bil, dtype=np.float32)
        raise ValueError("缺少 .hdr，无法确定 .bil/.bsq/.bip 的 lines/samples/bands")
    if tif is not None:
        import tifffile
        arr = tifffile.imread(tif)
        if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = arr.transpose(1, 2, 0)
        wavelengths = np.arange(arr.shape[2], dtype=float)
        return arr, wavelengths, {}
    if npy is not None:
        arr = np.load(npy)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        wavelengths = np.arange(arr.shape[2], dtype=float)
        return arr, wavelengths, {}
    raise FileNotFoundError(f"未找到高光谱数据: {path}")


def _read_pointcloud(path: Path) -> PointCloud:
    if path.is_dir():
        csv = next(path.glob("*.csv"), None)
        npy = next(path.glob("*.npy"), None)
        path = csv or npy
    if path is None or not path.exists():
        return PointCloud(xyz=np.zeros((0, 3)))
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(float)
    else:
        import pandas as pd
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        has_xyz = all(c in cols for c in ("x", "y", "z"))
        if not has_xyz:
            raise ValueError("点云 CSV 需包含 x,y,z 列")
        arr = df[["x", "y", "z"]].to_numpy(dtype=float)
        intensity = None
        if "reflectance" in cols:
            intensity = df["reflectance"].to_numpy(dtype=float)
        return PointCloud(xyz=arr, intensity=intensity)
    return PointCloud(xyz=arr[:, :3])


def _dn_ceiling(meta: Dict[str, Any]) -> Optional[float]:
    """判断是否整数 DN 数据，返回归一化除数（None 表示已是反射率/浮点）。

    规则：
      * 若 header 给出 ``ceiling``，用它（常见 12bit = 4095）；
      * 否则若 ``data type`` 是整型（uint16/int16...），用 2^bit_depth - 1 或 dtype 上限；
      * float32/64 视为已校正反射率，不归一。
    """
    dtype = meta.get("dtype")
    if dtype is None:
        return None
    bit_depth = meta.get("bit_depth")
    ceiling = meta.get("ceiling")
    if ceiling is not None:
        try:
            return float(str(ceiling).strip())
        except (TypeError, ValueError):
            pass
    if dtype in (np.float32, np.float64):
        return None
    if bit_depth is not None:
        try:
            return float(2 ** int(bit_depth) - 1)
        except (TypeError, ValueError):
            pass
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


class RealDatasetLoader(DataLoader):
    """从磁盘读取真实数据目录。目录约定：
+    "hyperspectral/" 或单个影像文件；"pointcloud/" 或点云文件；"labels.csv"。
    也可通过 ``cube_path``/``points_path``/``labels_path`` 显式指定。
    """

    name = "real"

    def __init__(
        self,
        root: Optional[str] = None,
        cube_path: Optional[str] = None,
        points_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        trait_columns: Optional[list[str]] = None,
        sample_id_column: str = "sample_id",
        max_pixels: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root) if root else None
        self.cube_path = Path(cube_path) if cube_path else None
        self.points_path = Path(points_path) if points_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.trait_columns = trait_columns or []
        self.sample_id_column = sample_id_column
        self.max_pixels = max_pixels        # 场景立方体的最大像素数（H*W）
        self.max_samples = max_samples      # SampleSet 最大样本数

    def _resolve(self, leaf: str) -> Optional[Path]:
        if self.root is None:
            return None
        direct = self.root / leaf
        if direct.exists():
            return direct
        # 先把无通配符的当作目录名/文件名递归找一次
        for pattern in (leaf, leaf):
            matches = list(self.root.glob(pattern))
            if not matches:
                matches = list(self.root.rglob(pattern))
            if matches:
                return matches[0]
        return None

    def load(self, **kwargs: Any) -> RealSource:
        cube_path = self.cube_path or self._resolve("hyperspectral") or self._resolve("*.hdr") \
            or self._resolve("*.tif") or self._resolve("*.npy")
        points_path = self.points_path or self._resolve("pointcloud") or self._resolve("points.csv")
        labels_path = self.labels_path or self._resolve("labels.csv") or self._resolve("trait*.csv")

        if cube_path is None or not cube_path.exists():
            raise FileNotFoundError("找不到高光谱数据，请检查数据目录/路径")
        if cube_path.is_dir():
            cube_path = next(cube_path.glob("*.hdr"), None) or next(cube_path.glob("*.tif"), None) \
                or next(cube_path.glob("*.npy"), None)

        cube, wavelengths, cube_meta = _read_cube(cube_path)

        # 场景降采样：控制内存/绘图开销（逐像素抽样，不做波段抽稀）。
        # 必须在整块归一化之前抽样，否则大扫描条带（如数百 MB ~ 数 GB）
        # 会先被 astype(float32) 整块放大，导致内存峰值过高。
        h, w, b = cube.shape
        if self.max_pixels and h * w > self.max_pixels:
            step = int(np.ceil(np.sqrt((h * w) / self.max_pixels)))
            cube = cube[::step, ::step, :]
            h, w = cube.shape[0], cube.shape[1]

        # DN(整数, 12/16bit) -> 反射率[0,1]：用 header 的 ceiling / 位深归一。
        # 若已有白板定标，保留原始 DN 交给 ReflectanceCorrection 处理。
        dn_ceiling = _dn_ceiling(cube_meta)
        if dn_ceiling is not None:
            cube = np.clip(cube.astype(np.float32) / dn_ceiling, 0.0, 1.0).astype(np.float32)
            dn_scaled = True
        else:
            cube = np.asarray(cube, dtype=np.float32)
            dn_scaled = False

        pointcloud = _read_pointcloud(points_path) if points_path else PointCloud(xyz=np.zeros((0, 3)))
        cal = Calibration(wavelengths=wavelengths)
        scene = RawScene(cube=cube, wavelengths=wavelengths, pointcloud=pointcloud, calibration=cal,
                         meta={"source": self.name, "dn_scaled": dn_scaled,
                               "dn_ceiling": dn_ceiling, "interleave": cube_meta.get("interleave"),
                               "bit_depth": cube_meta.get("bit_depth"),
                               "label": cube_meta.get("label", "")})

        # 构建 SampleSet：把立方体展平为逐像素样本，若存在标签则按像素/区域匹配
        X = cube.reshape(-1, b)
        y = None
        trait_names = None
        sample_ids = None
        if labels_path and labels_path.exists():
            import pandas as pd
            df = pd.read_csv(labels_path)
            trait_cols = [c for c in self.trait_columns if c in df.columns]
            trait_names = trait_cols or list(df.columns[1:4])
            y = df[trait_names].to_numpy(dtype=float)
            sample_ids = df[self.sample_id_column].astype(str).tolist() if self.sample_id_column in df.columns else None

        # 若标签样本数与像素不匹配，则取前 n 行作为样本级谱（实际使用时应按 ROI 聚合）
        if y is not None and y.shape[0] != X.shape[0]:
            n = min(y.shape[0], X.shape[0])
            X = X[:n]
            sample_ids = sample_ids[:n] if sample_ids else None

        # 样本抽样上限：避免十万级像素上 PCA/特征提取/可视化卡死
        if self.max_samples and X.shape[0] > self.max_samples:
            idx = np.linspace(0, X.shape[0] - 1, self.max_samples).astype(int)
            X = X[idx]
            if y is not None:
                y = y[idx]
            if sample_ids:
                sample_ids = [sample_ids[i] for i in idx]

        samples = SampleSet(
            X=X, wavelengths=wavelengths, y=y, trait_names=trait_names,
            sample_ids=sample_ids, meta={"source": self.name, "n_bands": b,
                                         "dn_scaled": dn_scaled, "dn_ceiling": dn_ceiling},
        )
        return RealSource(scene=scene, samples=samples)
