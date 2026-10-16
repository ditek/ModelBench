import time

import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import modelbench.bench_params as params
from modelbench.jobs import Job


class PyTorchRuntime:
    name = 'torch'

    def run(self, job: Job) -> list[float]:
        if job.precision != 'fp32':
            raise NotImplementedError(f'unsupported precision ({job.precision})')

        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
        torch.set_num_threads(job.threads)
        assert torch.get_num_threads() == job.threads
        array_input = np.random.default_rng(0).standard_normal(
            (1, 3, 224, 224), dtype=np.float32
        )
        tensor_input = torch.from_numpy(array_input)

        with torch.inference_mode():
            for _ in range(params.WARMUP):
                model(tensor_input)

        latencies = []
        with torch.inference_mode():
            for _ in range(params.REPEATS):
                t0 = time.perf_counter_ns()
                model(tensor_input)
                t1 = time.perf_counter_ns()
                latencies.append((t1 - t0) / 1e6)
        return latencies
