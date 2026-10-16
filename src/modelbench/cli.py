import argparse
import sys
import uuid
from collections import Counter
from pathlib import Path

from modelbench.adapters import DEVICE_KINDS, RUNTIMES
from modelbench.config import load_config
from modelbench.jobs import expand
from modelbench.orchestrator import run_jobs
from modelbench.results import Result
from modelbench.stats import summarize


def format_result(r: Result) -> str:
    head = f'{r.job.id} {r.job.runtime:<6} {r.job.precision:<5} t={r.job.threads:<2}'
    if r.status == 'ok':
        s = summarize(r.latencies_ms)
        return f'{head} ok      p50={s.p50:.2f} p95={s.p95:.2f} p99={s.p99:.2f} ms  {s.throughput:.0f}/s'
    return f'{head} {r.status:<7} {r.reason}'


def main():
    parser = argparse.ArgumentParser(prog='modelbench')
    sub = parser.add_subparsers(required=True, help='available commands')
    run = sub.add_parser('run')
    run.add_argument('config', type=Path)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--out', type=Path)
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
        print(format_result(r))
    if args.out:
        with args.out.open('w') as f:
            for r in results:
                f.write(r.model_dump_json() + '\n')
    counts = Counter(r.status for r in results)
    print(', '.join(f'{n} {status}' for status, n in sorted(counts.items())))

    if counts['failed']:
        sys.exit(1)
