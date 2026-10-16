import os
import platform

from modelbench import stats
from modelbench.jobs import Job


class FakeRuntime:
    name = 'fake'

    def run(self, job: Job) -> list[float]:
        return [1.0, 1.1, 1.2]


class LocalDevice:
    def __init__(self, name: str):
        self.name = name

    def probe(self) -> dict:
        import importlib.metadata as md

        info = {
            'logical_cpus': len(os.sched_getaffinity(0)),
            'cpu_model': stats.get_cpu_model(),
            'python_version': platform.python_version(),
            'machine': platform.machine(),
            'kernel': platform.release(),
        }
        try:
            # Those packages are not necessarily present
            versions = {
                'onnxruntime_version': md.version('onnxruntime'),
                'torch_version': md.version('torch'),
            }
            info.update(versions)
        except md.PackageNotFoundError:
            pass
        return info


def _ort():
    from modelbench.ort_runtime import OrtRuntime

    return OrtRuntime()


def _torch():
    from modelbench.torch_runtime import PyTorchRuntime

    return PyTorchRuntime()


DEVICE_KINDS = {'local': LocalDevice}
RUNTIMES = {'fake': FakeRuntime, 'ort': _ort, 'torch': _torch}
