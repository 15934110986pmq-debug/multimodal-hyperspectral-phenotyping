"""Stage 接口与 Pipeline 编排。
+
每个 stage 是 ``Stage`` 子类，实现 :meth:`run`。``Pipeline`` 顺序执行一组 stage，
并在 ``history`` 记录每个 stage 的执行时间与输出摘要，便于复现与出报告。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StageResult:
    """一个 stage 的执行结果：数据 + 摘要元信息。"""

    data: Any
    summary: Dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """流水线最小单元。"""

    name: str = "stage"

    def __call__(self, data: Any) -> StageResult:
        t0 = time.time()
        result = self.run(data)
        elapsed = time.time() - t0
        return StageResult(data=result, summary={"name": self.name, "elapsed_s": round(elapsed, 4)})

    @abstractmethod
    def run(self, data: Any) -> Any:
        """对输入数据做处理并返回新数据。"""

    def describe(self) -> str:
        return self.name


class Pipeline:
    """顺序执行一组 stage，保留每步摘要。"""

    def __init__(self, stages: List[Stage] | None = None) -> None:
        self.stages: List[Stage] = list(stages or [])
        self.history: List[Dict[str, Any]] = []

    def add(self, stage: Stage) -> "Pipeline":
        self.stages.append(stage)
        return self

    def run(self, data: Any) -> Any:
        self.history.clear()
        out = data
        for stage in self.stages:
            result = stage(out)
            out = result.data
            self.history.append(result.summary)
        return out

    def report(self) -> str:
        lines = [f"{h.get('name')}: {h.get('elapsed_s')}s" for h in self.history]
        return "\n".join(lines)
