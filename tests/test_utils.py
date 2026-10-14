from modelbench.jobs import Job


def make_job(**kw):
    base_job = {
        "model": "mobilenet_v2",
        "runtime": "fake",
        "device": "local",
        "precision": "fp32",
        "threads": 1,
    }
    return Job(**{**base_job, **kw})
