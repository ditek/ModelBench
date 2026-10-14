from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from modelbench.jobs import Job


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    job: Job
    status: Literal["ok", "skipped", "failed"]
    reason: str | None = None
    latencies_ms: list[float] = Field(default_factory=list)
    device_metadata: dict | None = None
