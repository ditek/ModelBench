import argparse
import sys
import uuid
from collections import Counter
from pathlib import Path

from modelbench.adapters import DEVICE_KINDS, RUNTIMES, LocalDevice
from modelbench.config import DeviceConfig, load_config
from modelbench.jobs import Job, expand
from modelbench.results import Result
from modelbench.stats import summarize


def format_result(r: Result) -> str:
    head = f'{r.job.id} {r.job.runtime:<6} {r.job.precision:<5} t={r.job.threads:<2}'
    if r.status == 'ok':
        s = summarize(r.latencies_ms)
        return f'{head} ok      p50={s.p50:.2f} p95={s.p95:.2f} p99={s.p99:.2f} ms  {s.throughput:.0f}/s'
    return f'{head} {r.status:<7} {r.reason}'


def handle_run(args: argparse.Namespace):
    conf = load_config(args.config)
    jobs = expand(conf)
    if args.dry_run:
        print(f'{len(jobs)} jobs configured')
        for job in jobs:
            print(
                f'Job {job.id}: {job.model}, {job.runtime}, {job.device}, {job.precision}, {job.threads}'
            )
        return
    run_id = uuid.uuid4().hex[:8]
    devices = {x.name: DEVICE_KINDS[x.kind](x) for x in conf.devices}
    unknown = {j.runtime for j in jobs} - RUNTIMES.keys()
    if unknown:
        print(f'Found unknown runtimes: {unknown}')
        sys.exit(1)

    results: list[Result] = []
    for dev in devices.values():
        dev_jobs = [j for j in jobs if j.device == dev.name]
        results.extend(dev.run(dev_jobs, run_id))
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


def handle_worker(args: argparse.Namespace):
    run_id = args.run_id if args.run_id else uuid.uuid4().hex[:8]
    for line in sys.stdin:
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            job = Job.model_validate_json(line)
        except Exception as e:
            print(e.args, file=sys.stderr)
            sys.exit(1)
        device = LocalDevice(DeviceConfig(name=job.device, kind='local'))
        result = device.run([job], run_id=run_id)[0]
        print(result.model_dump_json(), flush=True)


def main():
    parser = argparse.ArgumentParser(prog='modelbench')
    sub = parser.add_subparsers(required=True, help='available commands')

    # 'run' command
    run = sub.add_parser('run')
    run.add_argument('config', type=Path)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--out', type=Path)
    run.set_defaults(func=handle_run)

    # 'worker' command
    worker = sub.add_parser(
        'worker',
        help='Input job config. Each line is one Job as JSON. Enter an empty line to abort.',
    )
    worker.add_argument('--run-id', type=str)
    worker.set_defaults(func=handle_worker)

    args = parser.parse_args()
    args.func(args)
