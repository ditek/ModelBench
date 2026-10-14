import os

from modelbench.jobs import Job


class FakeRuntime:
    name = "fake"

    def run(self, job: Job) -> list[float]:
        return [1.0, 1.1, 1.2]


class LocalDevice:
    def __init__(self, name: str):
        self.name = name

    def probe(self) -> dict:
        return {"cores": len(os.sched_getaffinity(0))}
