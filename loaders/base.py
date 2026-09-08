"""数据集加载器接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from data.schema import SampleSet


class DataLoader(ABC):
    """任何数据源都要实现 :meth:`load`，返回统一 :class:`SampleSet`。"""

    name: str = "loader"

    @abstractmethod
    def load(self, **kwargs: Any) -> SampleSet:
        """读取数据并转换为统一内部格式。"""
