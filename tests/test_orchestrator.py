from test_utils import make_job

from modelbench.adapters import FakeRuntime, LocalDevice
from modelbench.orchestrator import run_jobs


def test_run_jobs_ok():
    job = make_job()
    device = LocalDevice(job.device)
    runtime = FakeRuntime()
    run_id = "test_run_1"
    results = run_jobs([job], device, runtime, run_id)

    assert len(results) == 1
    result = results[0]

    assert result.run_id == run_id
    assert result.job == job
    assert result.status == "ok"
    assert result.latencies_ms == [1.0, 1.1, 1.2]
    assert result.device_metadata is not None


def test_run_jobs_skipped():
    job = make_job(threads=99)
    device = LocalDevice(job.device)
    runtime = FakeRuntime()
    run_id = "test_run_2"
    results = run_jobs([job], device, runtime, run_id)

    assert len(results) == 1
    result = results[0]
    assert result.status == "skipped"
    assert len(result.latencies_ms) == 0
