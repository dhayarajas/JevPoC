"""Arm B: a constrained LLM answering the same catalogue questions (deck slide 46, System One adapter style).

With LLM_API_KEY and LLM_API_URL set it calls an OpenAI-compatible chat endpoint in JSON mode.
Without them it uses an offline stand-in: the mock with a different seed, label-only nouls and
verbalised (always 1.0) choice confidence, plus typical frontier-LLM latency and price.
The stand-in tests the protocol; it says nothing about any real LLM.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from jevpoc.client import JevError, JevResponse, JevTimeout
from jevpoc.mock import MockJev

LLM_PRICE_IN_PER_MTOK = 2.50
LLM_PRICE_OUT_PER_MTOK = 10.0
PROMPT_OVERHEAD_TOKENS = 350  # system prompt and JSON schema


@dataclass
class LLMResponse(JevResponse):
    @property
    def cost_usd(self) -> float:
        u = self.usage
        return (
            u.get("input_tokens", 0) * LLM_PRICE_IN_PER_MTOK + u.get("output_tokens", 0) * LLM_PRICE_OUT_PER_MTOK
        ) / 1_000_000


def _labels_only(answers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for qid, a in answers.items():
        if a["type"] == "noul":
            out[qid] = {"type": "noul", "noul": 1.0 if a["noul"] >= 0.5 else 0.0}
        elif a["type"] == "choice":
            out[qid] = {"type": "choice", "choice": a["choice"], "confidence": 1.0}
        else:
            out[qid] = {"type": "score", "score": float(round(a["score"])), "confidence": 1.0}
    return out


class StandInLLM:
    backend = "llm-mock"

    def __init__(self, model: str = "constrained-llm-standin"):
        self.model = model
        self._mock = MockJev(model)

    def ask(self, state, questions, model=None, timeout_s: float = 5.0, request_id=None) -> LLMResponse:
        r = self._mock.ask(state, questions, self.model, timeout_s=10.0)
        tin = r.input_tokens + PROMPT_OVERHEAD_TOKENS
        usage = {"input_tokens": tin, "output_tokens": 12 * len(questions)}
        latency = 650 + 3 * r.latency_ms
        return LLMResponse(self.model, _labels_only(r.answers), usage, latency, self.backend, {})


def _prompt(questions: dict[str, dict[str, Any]]) -> str:
    spec = {
        qid: {
            "type": q["type"],
            "instructions": q["instructions"],
            "allowed": list(q["criteria"]) if q["type"] != "noul" else ["true", "false"],
        }
        for qid, q in questions.items()
    }
    return (
        "Answer each question about the state. Reply with JSON only: "
        '{"<id>": {"answer": <one allowed value or score index>}}. Questions: ' + json.dumps(spec)
    )


class OpenAICompatLLM:
    backend = "llm"

    def __init__(self, url: str, api_key: str, model: str, transport: httpx.BaseTransport | None = None):
        self.url, self.model = url.rstrip("/"), model
        self._http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def ask(self, state, questions, model=None, timeout_s: float = 5.0, request_id=None) -> LLMResponse:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _prompt(questions)},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
        }
        t0 = time.perf_counter()
        try:
            r = self._http.post(f"{self.url}/chat/completions", json=body, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise JevTimeout(str(e)) from e
        except httpx.HTTPError as e:
            raise JevError(str(e)) from e
        if r.status_code != 200:
            raise JevError(f"HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        try:
            raw = json.loads(d["choices"][0]["message"]["content"])
            answers = {}
            for qid, q in questions.items():
                v = raw[qid]["answer"]
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": 1.0 if str(v).lower() == "true" else 0.0}
                elif q["type"] == "choice":
                    if v not in q["criteria"]:
                        raise JevError(f"{qid}: {v!r} is not an allowed choice")
                    answers[qid] = {"type": "choice", "choice": v, "confidence": 1.0}
                else:
                    answers[qid] = {"type": "score", "score": float(v), "confidence": 1.0}
        except (KeyError, ValueError, TypeError) as e:
            raise JevError(f"unparseable LLM answer: {e}") from e
        u = d.get("usage", {})
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
        return LLMResponse(self.model, answers, usage, (time.perf_counter() - t0) * 1000, self.backend, d)


def get_llm_client():
    key, url = os.environ.get("LLM_API_KEY"), os.environ.get("LLM_API_URL")
    if key and url:
        return OpenAICompatLLM(url, key, os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    return StandInLLM()
