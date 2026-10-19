from modelbench.jobs import Job
from modelbench.results import Result


def run_jobs(jobs: list[Job], device, runtime, run_id: str) -> list[Result]:
    meta = device.probe()
    results = []
    for job in jobs:
        if job.threads > meta['logical_cpus']:
            results.append(
                Result(
                    run_id=run_id,
                    job=job,
                    status='skipped',
                    reason=f'requested {job.threads} threads, device has {meta["logical_cpus"]} logical CPU(s)',
                    device_metadata=meta,
                )
            )
            continue
        try:
            latencies = runtime.run(job)
            results.append(
                Result(
                    run_id=run_id,
                    job=job,
                    status='ok',
                    latencies_ms=latencies,
                    device_metadata=meta,
                )
            )
        except Exception as e:
            results.append(
                Result(
                    run_id=run_id,
                    job=job,
                    status='failed',
                    reason=str(e),
                    device_metadata=meta,
                )
            )
    return results
