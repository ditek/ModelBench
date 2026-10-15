import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from modelbench.jobs import Job

WARMUP = 10
REPEATS = 100


class OrtRuntime:
    name = 'ort'

    def __init__(self, model_path=Path('models/mobilenet_v2.onnx')):
        self.model_path = model_path

    def run(self, job: Job) -> list[float]:
        if job.precision != 'fp32':
            raise NotImplementedError(f'unsupported precision ({job.precision})')

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = job.threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            str(self.model_path), opts, providers=['CPUExecutionProvider']
        )
        input_name = session.get_inputs()[0].name
        x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)

        for _ in range(WARMUP):
            session.run(None, {input_name: x})

        latencies = []
        for _ in range(REPEATS):
            t0 = time.perf_counter_ns()
            session.run(None, {input_name: x})
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)
        return latencies
