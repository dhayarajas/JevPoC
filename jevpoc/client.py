"""Thin client for POST /v1/systemone, plus a backend switch.

Real Jev is used when TYPESAFE_API_KEY is set. Otherwise a local mock answers
with the same response shape so every level runs offline. Mock answers are not
Jev answers and must never be quoted as results.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

API_URL = os.environ.get("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")
DEFAULT_MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
PRICE_PER_MTOK = 0.042


class JevError(RuntimeError):
    """A failed call. Callers must treat this as 'no answer', never as 'no'."""


class JevTimeout(JevError):
    pass


@dataclass
class JevResponse:
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    latency_ms: float
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return int(self.usage.get("input_tokens", 0))

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_MTOK / 1_000_000

    def choice(self, qid: str) -> str:
        return self.answers[qid]["choice"]

    def noul(self, qid: str) -> float:
        return float(self.answers[qid]["noul"])

    def score(self, qid: str) -> float:
        return float(self.answers[qid]["score"])

    def confidence(self, qid: str) -> float:
        a = self.answers[qid]
        if "confidence" in a:
            return float(a["confidence"])
        return abs(2 * float(a["noul"]) - 1)


def build_request(state: Any, questions: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    return {"model": model, "state": state, "questions": questions}


class HttpJev:
    backend = "jev"

    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL, url: str = API_URL, transport: httpx.BaseTransport | None = None
    ):
        self.model = model
        self.url = url
        self._http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def ask(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        model: str | None = None,
        timeout_s: float = 5.0,
    ) -> JevResponse:
        body = build_request(state, questions, model or self.model)
        t0 = time.perf_counter()
        try:
            r = self._http.post(self.url, json=body, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise JevTimeout(f"timeout after {timeout_s * 1000:.0f} ms") from e
        except httpx.HTTPError as e:
            raise JevError(str(e)) from e
        latency = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise JevError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return JevResponse(data["model"], data["answers"], data.get("usage", {}), latency, self.backend, data)


def get_client(model: str | None = None):
    """Real Jev if TYPESAFE_API_KEY is set (and JEV_BACKEND != mock), else the mock."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key and os.environ.get("JEV_BACKEND", "").lower() != "mock":
        return HttpJev(key, model or DEFAULT_MODEL)
    from jevpoc.mock import MockJev

    return MockJev(model or DEFAULT_MODEL)


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
