import os
import platform
import shlex
import subprocess
import sys

from pydantic import ValidationError

from modelbench import stats
from modelbench.config import DeviceConfig
from modelbench.jobs import Job
from modelbench.orchestrator import run_jobs
from modelbench.results import Result


class FakeRuntime:
    name = 'fake'

    def run(self, job: Job) -> list[float]:
        return [1.0, 1.1, 1.2]


class LocalDevice:
    def __init__(self, cfg: DeviceConfig):
        self.name = cfg.name

    def probe(self) -> dict:
        import importlib.metadata as md

        info = {
            'logical_cpus': len(os.sched_getaffinity(0)),
            'cpu_model': stats.get_cpu_model(),
            'python_version': platform.python_version(),
            'machine': platform.machine(),
            'kernel': platform.release(),
        }
        # Those packages are not necessarily present
        for pkg in ('onnxruntime', 'torch'):
            try:
                info[f'{pkg}_version'] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
        return info

    def run(self, jobs: list[Job], run_id: str) -> list[Result]:
        results = []
        for job in jobs:
            try:
                runtime = RUNTIMES[job.runtime]()
            except KeyError:
                results.append(
                    Result(
                        run_id=run_id,
                        job=job,
                        status='failed',
                        reason=f'Runtime {job.runtime} is invalid',
                    )
                )
                continue
            except Exception as e:
                results.append(
                    Result(
                        run_id=run_id,
                        job=job,
                        status='failed',
                        reason=f'{type(e).__name__}: {e}',
                    )
                )
                continue
            results.extend(run_jobs([job], self, runtime, run_id))
        return results


class SshDevice:
    def __init__(self, cfg: DeviceConfig):
        self.name = cfg.name
        self.host = cfg.host
        self.remote_dir = cfg.remote_dir
        self.uv_path = cfg.uv_path

    def run(self, jobs: list[Job], run_id: str) -> list[Result]:
        inputs = '\n'.join(job.model_dump_json() for job in jobs) + '\n'
        timeout_s = 90 + 20 * len(jobs)
        stdout, error = '', None
        try:
            q = shlex.quote
            remote = f'cd {q(str(self.remote_dir))} && {q(self.uv_path)} run --locked --extra bench modelbench worker --run-id {q(run_id)}'
            cmd = [
                'ssh',
                '-o',
                'BatchMode=yes',
                '-o',
                'ConnectTimeout=10',
                self.host,
                remote,
            ]
            ret = subprocess.run(
                cmd,
                check=False,
                input=inputs,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout = ret.stdout
            if ret.returncode != 0:
                error = f'ssh exited with {ret.returncode}: {ret.stderr[-300:]}'
        except subprocess.TimeoutExpired:
            error = f'timed out after {timeout_s}s'
        except OSError as e:
            error = f'could not start ssh: {e}'

        results = []
        for line in stdout.split(sep='\n'):
            line = line.strip()
            if not line:
                continue
            try:
                r = Result.model_validate_json(line)
            except ValidationError as e:
                print(e.errors(), line[:200], file=sys.stderr)
                continue
            results.append(r)
        result_map = {r.job.id: r for r in results}
        reason = error or 'no result returned from worker'
        return [
            result_map.get(j.id)
            or Result(run_id=run_id, job=j, status='failed', reason=reason)
            for j in jobs
        ]


def _ort():
    from modelbench.ort_runtime import OrtRuntime

    return OrtRuntime()


def _torch():
    from modelbench.torch_runtime import PyTorchRuntime

    return PyTorchRuntime()


DEVICE_KINDS = {'local': LocalDevice, 'ssh': SshDevice}
RUNTIMES = {'fake': FakeRuntime, 'ort': _ort, 'torch': _torch}
