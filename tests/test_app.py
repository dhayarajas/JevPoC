from fastapi.testclient import TestClient

from jevpoc.gateway.app import create_app
from jevpoc.usecases import care_intent


def test_endpoints(gateway):
    c = TestClient(create_app(gateway))
    assert c.get("/").status_code == 200
    assert c.get("/v1/health").json()["backend"] == "mock"
    assert len(c.get("/v1/catalogue").json()) == 7
    r = c.post(f"/v1/decide/{care_intent.NAME}", json={"context": care_intent.EXAMPLES[1]})
    assert r.status_code == 200 and r.json()["action"]
    assert c.post("/v1/decide/nope", json={"context": {}}).status_code == 404
    assert len(c.get("/v1/decisions").json()) == 1
    assert c.post("/v1/chaos", json={"fail_rate": 1}).json()["chaos"]["fail_rate"] == 1
    d = c.post(f"/v1/decide/{care_intent.NAME}", json={"context": care_intent.EXAMPLES[1]}).json()
    assert d["fallback_used"]
