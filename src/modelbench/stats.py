from dataclasses import dataclass

import numpy as np


@dataclass
class ResultSummary:
    p50: float
    p95: float
    p99: float
    avg: float
    min: float
    max: float
    throughput: float


def summarize(latencies: list[float]) -> ResultSummary:
    avg = float(np.average(latencies))
    return ResultSummary(
        p50=float(np.percentile(latencies, 50)),
        p95=float(np.percentile(latencies, 95)),
        p99=float(np.percentile(latencies, 99)),
        avg=avg,
        min=min(latencies),
        max=max(latencies),
        throughput=1000 / avg,
    )
