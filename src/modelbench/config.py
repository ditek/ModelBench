from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DeviceConfig(BaseModel):
    name: str
    kind: Literal['local']


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
