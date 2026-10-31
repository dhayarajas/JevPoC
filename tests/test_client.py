import json

import httpx
import pytest

from jevpoc.client import HttpJev, JevError, JevResponse, JevTimeout, build_request

QS = {"q": {"type": "noul", "instructions": "x"}}


def test_request_shape_matches_docs():
    body = build_request({"a": 1}, QS, "jev-1.14.0")
    assert body == {"model": "jev-1.14.0", "state": {"a": 1}, "questions": QS}


def test_http_client_sends_bearer_and_parses():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {"q": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 1000},
            },
        )

    c = HttpJev("sk-test", transport=httpx.MockTransport(handler))
    r = c.ask({"a": 1}, QS)
    assert seen["auth"] == "Bearer sk-test"
    assert set(seen["body"]) == {"model", "state", "questions"}
    assert r.noul("q") == 0.9
    assert r.confidence("q") == pytest.approx(0.8)
    assert r.cost_usd == pytest.approx(0.042 / 1000)


def test_http_errors_are_errors_not_answers():
    c = HttpJev("k", transport=httpx.MockTransport(lambda req: httpx.Response(529, text="overloaded")))
    with pytest.raises(JevError, match="529"):
        c.ask({}, QS)

    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(JevTimeout):
        HttpJev("k", transport=httpx.MockTransport(slow)).ask({}, QS)


def test_choice_confidence_comes_from_response():
    r = JevResponse("m", {"c": {"type": "choice", "choice": "a", "confidence": 0.4}}, {}, 1, "jev")
    assert r.confidence("c") == 0.4
