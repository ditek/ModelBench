import argparse
import uuid
from pathlib import Path

from modelbench.adapters import DEVICE_KINDS, RUNTIMES
from modelbench.config import load_config
from modelbench.jobs import expand
from modelbench.orchestrator import run_jobs
from modelbench.results import Result


def main():
    parser = argparse.ArgumentParser(prog='modelbench')
    sub = parser.add_subparsers(required=True, help='available commands')
    run = sub.add_parser('run')
    run.add_argument('config', type=Path)
    run.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    conf = load_config(args.config)
    jobs = expand(conf)
    if args.dry_run:
        print(f'{len(jobs)} jobs configured')
        for job in jobs:
            print(
                f'Job {job.id}: {job.model}, {job.runtime}, \
                  {job.device}, {job.precision}, {job.threads}'
            )
        return
    run_id = uuid.uuid4().hex[:8]
    devices = {x.name: DEVICE_KINDS[x.kind](x.name) for x in conf.devices}
    runtimes = {x: RUNTIMES[x]() for x in conf.runtimes}

    results: list[Result] = []
    for dev_name, dev in devices.items():
        for rt_name, rt in runtimes.items():
            jobs_subset = [
                job for job in jobs if job.device == dev_name and job.runtime == rt_name
            ]
            results.extend(run_jobs(jobs_subset, dev, rt, run_id))

    if len(results) != len(jobs):
        raise ValueError(
            f'Number of results ({len(results)}) does not match number of jobs ({len(jobs)})'
        )

    for r in results:
        print(r.job.id, r.status, r.reason or '')
