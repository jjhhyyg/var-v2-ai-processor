"""Lightweight timing aggregation for desktop task diagnostics."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional


STAGE_LABELS = {
    "videoRead": "视频读取",
    "detectorTotal": "检测总耗时",
    "detectorPreprocess": "检测预处理",
    "detectorInference": "模型推理",
    "detectorPostprocess": "检测后处理",
    "detectorNms": "NMS",
    "metrics": "动态指标",
    "progressEmit": "进度通知",
    "eventGeneration": "事件生成",
    "detectionsPersist": "检测结果持久化",
    "resultEmit": "结果提交",
    "resultVideoExport": "结果视频导出",
}

PERCENT_BASE_STAGES = {
    "videoRead",
    "detectorTotal",
    "metrics",
    "progressEmit",
    "eventGeneration",
    "detectionsPersist",
    "resultEmit",
    "resultVideoExport",
}


class PerformanceTrace:
    """Collect stage timings as aggregate samples only."""

    def __init__(self) -> None:
        self._samples_ms: Dict[str, List[float]] = {}

    def record_elapsed_ns(self, stage: str, elapsed_ns: int) -> None:
        elapsed_ms = max(0.0, elapsed_ns / 1_000_000.0)
        self._samples_ms.setdefault(stage, []).append(elapsed_ms)

    def record_elapsed_since(self, stage: str, start_ns: int) -> None:
        self.record_elapsed_ns(stage, time.perf_counter_ns() - start_ns)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_elapsed_since(stage, start_ns)

    def summary(self, total_measured_frames: int) -> Dict[str, object]:
        base_total_ms = sum(
            sum(values)
            for stage, values in self._samples_ms.items()
            if stage in PERCENT_BASE_STAGES
        )
        if base_total_ms <= 0:
            base_total_ms = sum(sum(values) for values in self._samples_ms.values())

        stages = []
        for stage in STAGE_LABELS:
            values = self._samples_ms.get(stage)
            if not values:
                continue
            total_ms = sum(values)
            stages.append(
                {
                    "key": stage,
                    "label": STAGE_LABELS[stage],
                    "samples": len(values),
                    "totalMs": round(total_ms, 3),
                    "avgMs": round(total_ms / len(values), 3),
                    "p50Ms": round(self._percentile(values, 50), 3),
                    "p95Ms": round(self._percentile(values, 95), 3),
                    "maxMs": round(max(values), 3),
                    "percentOfMeasuredMs": round((total_ms / base_total_ms) * 100, 3) if base_total_ms > 0 else 0.0,
                }
            )

        return {
            "schemaVersion": 1,
            "totalMeasuredFrames": int(total_measured_frames),
            "stages": stages,
        }

    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = max(0, math.ceil((percentile / 100.0) * len(sorted_values)) - 1)
        return sorted_values[index]


def record_elapsed_since(trace: Optional[PerformanceTrace], stage: str, start_ns: int) -> None:
    if trace is not None:
        trace.record_elapsed_since(stage, start_ns)
