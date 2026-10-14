from test_utils import make_job

from modelbench.config import Config, DeviceConfig
from modelbench.jobs import expand


def test_id_is_stable():
    assert make_job().id == make_job().id


def test_id_changes_with_field():
    assert make_job().id != make_job(threads=2).id


def test_id_format():
    assert len(make_job().id) == 12


def test_expand():
    conf = Config(
        models=['mobilenet_v2'],
        runtimes=['fake', 'real'],
        devices=[
            DeviceConfig(name='local', kind='local'),
            DeviceConfig(name='local2', kind='local'),
        ],
        precisions=['fp32', 'int8'],
        threads=[1, 2, 4],
    )
    jobs = expand(conf)
    assert len({j.id for j in jobs}) == 24  # all distinct
    assert expand(conf) == expand(conf)  # deterministic order
