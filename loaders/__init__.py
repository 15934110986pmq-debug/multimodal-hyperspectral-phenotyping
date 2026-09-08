"""数据集加载器：统一输出 :class:`~data.schema.SampleSet`。"""

from .base import DataLoader
from .synthetic import SyntheticLoader
from .greenhyperspectra import GreenHyperSpectraLoader
from .real import RealDatasetLoader, RealSource

LOADERS = {
    "synthetic": SyntheticLoader,
    "greenhyperspectra": GreenHyperSpectraLoader,
    "real": RealDatasetLoader,
}

__all__ = ["DataLoader", "SyntheticLoader", "GreenHyperSpectraLoader",
           "RealDatasetLoader", "RealSource", "LOADERS"]
