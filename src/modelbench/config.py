from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeviceConfig(BaseModel):
    name: str
    kind: Literal['local', 'ssh']
    host: str | None = None
    remote_dir: str | None = None
    uv_path: str = 'uv'

    @model_validator(mode='after')
    def _ssh_needs_host(self):
        if self.kind == 'ssh' and not (self.host and self.remote_dir):
            raise ValueError("ssh devices require 'host' and 'remote_dir'")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    models: list[str] = Field(min_length=1)
    runtimes: list[str] = Field(min_length=1)
    devices: list[DeviceConfig] = Field(min_length=1)
    precisions: list[Literal['fp32', 'int8']] = Field(min_length=1)
    threads: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())
    return Config.model_validate(data)
