import os

import pytest

os.environ["JEV_BACKEND"] = "mock"
os.environ.pop("JEV_MOCK_FAIL_RATE", None)
os.environ.pop("JEV_MOCK_SLOW_RATE", None)


@pytest.fixture
def gateway(tmp_path):
    from jevpoc.gateway.core import DecisionLog, Gateway
    from jevpoc.mock import MockJev

    return Gateway(client=MockJev("jev-1.14.0"), log=DecisionLog(tmp_path / "log.sqlite"))
