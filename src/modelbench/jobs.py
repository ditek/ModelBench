import hashlib
import itertools
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from modelbench.config import Config


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    runtime: str
    device: str
    precision: Literal["fp32", "int8"]
    threads: int = Field(gt=0)

    @property
    def id(self) -> str:
        d = self.model_dump_json()
        s = json.dumps(d, sort_keys=True)
        print(s)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def expand(conf: Config) -> list[Job]:
    return [
        Job(model=m, runtime=r, device=d.name, precision=p, threads=t)
        for d, m, r, p, t in itertools.product(
            conf.devices, conf.models, conf.runtimes, conf.precisions, conf.threads
        )
    ]
