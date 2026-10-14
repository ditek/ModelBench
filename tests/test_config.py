import copy

import pytest
import yaml
from pydantic import ValidationError

from modelbench.config import Config, load_config

VALID = {
    'models': ['mobilenet_v2'],
    'runtimes': ['fake'],
    'devices': [{'name': 'local', 'kind': 'local'}],
    'precisions': ['fp32', 'int8'],
    'threads': [1, 2, 4],
}


def make(**overrides):
    data = copy.deepcopy(VALID)
    data.update(overrides)
    return data


def test_valid_config_parses():
    cfg = Config.model_validate(VALID)
    assert cfg.threads == [1, 2, 4]
    assert cfg.precisions == ['fp32', 'int8']


@pytest.mark.parametrize(
    'bad',
    [
        pytest.param(make(thread=[1]), id='extra-key'),
        pytest.param(make(models=[]), id='empty-list'),
        pytest.param(make(threads=[0]), id='zero-threads'),
        pytest.param(make(precisions=['fp64']), id='unknown-precision'),
    ],
)
def test_invalid_config_rejected(bad):
    with pytest.raises(ValidationError):
        Config.model_validate(bad)


def test_load_config_from_file(tmp_path):
    p = tmp_path / 'cfg.yaml'
    p.write_text(yaml.safe_dump(VALID))
    assert load_config(p).models == ['mobilenet_v2']
