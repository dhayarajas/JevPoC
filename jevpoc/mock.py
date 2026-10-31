"""Offline stand-in for Jev with the same request and response shapes.

It reads cue phrases and a few structured fields, adds seeded noise, and
returns probabilities. It exists so the pipeline, policies, log and eval
harness run without a key. Its numbers say nothing about Jev's accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from typing import Any

from jevpoc.client import JevError, JevResponse, JevTimeout, build_request

# Cue phrases per question id and option (choice) or polarity (noul).
CHOICE_CUES: dict[str, dict[str, list[str]]] = {
    "language": {
        "es": ["hola", "quiero", "factura", "por favor", "mi ", "no funciona", "gracias", "cuenta", "señal", "¿"],
        "en": [" the ", " my ", " is ", " i ", "i'm", " and ", " you", "please", "what", "how"],
        "other": ["bonjour", "merci", "obrigado", "danke", "xin chào"],
    },
    "department": {
        "billing": ["charge", "bill", "refund", "invoice", "subscription"],
        "technical": ["error", "500", "crash", "integration", "bug", "not working"],
        "sales": ["pricing", "price", "quote", "upgrade my plan"],
    },
    "intent": {
        "billing_question": [
            "amount",
            "invoice",
            "line item",
            "went up",
            "bill",
            "charged",
            "charge",
            "fee",
            "higher than",
            "overcharged",
            "statement",
        ],
        "payment_arrangement": [
            "more time to pay",
            "get paid",
            "move my due",
            "pay later",
            "extension",
            "payment arrangement",
            "can't pay",
            "due date",
            "split",
        ],
        "plan_change": [
            "plan",
            "perk",
            "lower monthly",
            "fewer features",
            "change my plan",
            "switch plan",
            "cheaper plan",
            "add a line",
            "downgrade",
            "unlimited",
        ],
        "device_tech_support": [
            "drains",
            "update",
            "camera",
            "won't",
            "won't turn on",
            "screen",
            "battery",
            "frozen",
            "restart",
            "apps crash",
            "wifi calling",
        ],
        "network_coverage": [
            "one bar",
            "towers",
            "commute",
            "nothing loads",
            "dropping",
            "no signal",
            "dropped calls",
            "no service",
            "bars",
            "slow data",
            "outage",
            "coverage",
        ],
        "activation_sim": [
            "set up",
            "new line",
            "watch",
            "activate",
            "new sim",
            "esim",
            "sim card",
            "transfer my number",
            "activation",
        ],
        "order_status": [
            "arrived",
            "accessory",
            "where is my",
            "order",
            "shipped",
            "tracking",
            "delivery",
            "hasn't arrived",
        ],
        "international_roaming": [
            "overseas",
            "traveling to",
            "going on a trip",
            "travel",
            "abroad",
            "roaming",
            "international",
            "trip",
            "travelpass",
        ],
        "cancel_service": [
            "another carrier",
            "to leave",
            "cancel",
            "leave",
            "switching to",
            "close my account",
            "port out",
            "done with",
        ],
        "fraud_unrecognised": [
            "didn't add",
            "was not me",
            "changed my sim",
            "didn't make",
            "don't recognize",
            "unrecognized",
            "someone else",
            "hacked",
            "not me",
        ],
        "device_upgrade": ["pixel", "iphone", "deals", "upgrade", "new phone", "trade in", "trade-in", "eligible for"],
    },
    "failure_type": {
        "invented_credit": ["credit", "refund", "waive", "discount"],
        "wrong_price": ["$", "per month", "/mo"],
        "wrong_plan_details": ["unlimited", "hotspot", "gb", "5g"],
        "wrong_policy": ["days", "return", "fee", "policy"],
    },
    "fault_domain": {
        "ran": ["cell", "sector", "radio", "vswr", "enodeb", "gnodeb", "antenna", "rru"],
        "transport": ["backhaul", "fiber", "link down", "microwave", "los", "router"],
        "core": ["mme", "amf", "smf", "upf", "hss", "registration", "core"],
        "power": ["power", "battery", "rectifier", "generator", "mains", "ac fail"],
        "customer_premises": ["router at home", "cpe", "fios box", "ont", "customer premises", "single customer"],
    },
    "approval_level": {
        "none": ["read", "get", "list", "describe", "query", "test"],
        "peer": ["update", "restart", "scale"],
        "owner": ["prod", "production", "customer data", "billing"],
        "block": ["delete", "drop", "truncate", "disable", "credit", "refund", "wipe"],
    },
    "recommended_action": {
        "allow": ["in store", "id verified", "long tenure"],
        "step_up": ["new device", "sms_otp", "web"],
        "hold": ["urgent", "port_out", "pressure", "traveling", "changes_72h_2_plus"],
    },
}

NOUL_CUES: dict[str, tuple[list[str], list[str]]] = {
    "is_urgent": (["urgent", "asap", "can't process", "down", "minutes ago", "immediately"], ["whenever", "no rush"]),
    "needs_human": (
        ["speak to a person", "real person", "human", "agent", "representative", "supervisor", "lawyer"],
        [],
    ),
    "in_scope": (
        ["phone", "bill", "plan", "line", "sim", "order", "signal", "service", "account", "device", "data", "charge"],
        ["weather", "recipe", "football", "poem"],
    ),
    "churn_language": (["cancel", "switch to", "leaving", "t-mobile", "at&t", "done with", "close my account"], []),
    "bot_resolvable": (
        ["how do i", "where can i", "what is", "reset", "track"],
        ["third time", "nobody", "still", "again", "refund"],
    ),
    "asks_for_human": (["real person", "human", "agent", "supervisor", "representative"], []),
    "open_to_offer": (
        ["looking for", "interested", "better deal", "cheaper", "what can you do", "options"],
        ["just fix", "not interested", "no offers"],
    ),
    "truck_roll": (
        ["power", "fiber cut", "hardware", "rru", "antenna", "physical", "generator"],
        ["software", "config", "reboot cleared"],
    ),
    "change_related": (["after change", "maintenance window", "cr-", "software upgrade", "config push"], []),
    "pressure_tactics": (
        ["urgent", "right now", "traveling", "boss", "supervisor", "lost my phone", "hurry", "don't have time"],
        ["in store", "calm"],
    ),
    "story_consistent": (
        ["lost", "broken", "new phone", "upgraded"],
        ["doesn't know", "couldn't answer", "wrong pin", "hesitated"],
    ),
    "on_goal": (["goal", "requested"], ["unrelated"]),
    "looping": (["retry", "again", "same call"], []),
    "answers_question": (["you can", "your", "the fee", "yes", "no"], []),
}

SCORE_CUES: dict[str, list[tuple[str, float]]] = {
    "frustration": [
        ("calm", -1.0),
        ("thanks", -0.6),
        ("please", -0.3),
        ("again", 0.8),
        ("third time", 1.5),
        ("ridiculous", 1.5),
        ("frustrated", 1.4),
        ("unacceptable", 1.8),
        ("cancel", 1.6),
        ("!!", 1.0),
        ("still", 0.7),
        ("nobody", 1.0),
        ("waste", 1.2),
        ("switch to", 1.8),
        ("angry", 1.6),
    ],
    "severity": [
        ("major", 1.5),
        ("critical", 2.0),
        ("down", 1.4),
        ("outage", 1.6),
        ("multiple sites", 1.8),
        ("minor", -1.0),
        ("warning", -0.6),
        ("cleared", -1.5),
        ("single customer", -1.2),
        ("hospital", 1.2),
    ],
    "save_risk": [("cancel", 1.6), ("switch to", 1.8), ("too expensive", 1.0), ("competitor", 1.2), ("happy", -1.2)],
}

CustomFn = Callable[[Any, str, dict[str, Any], random.Random], dict[str, Any]]


def _text(state: Any) -> str:
    return json.dumps(state, ensure_ascii=False).lower() if not isinstance(state, str) else state.lower()


def _hits(text: str, cues: list[str]) -> int:
    return sum(1 for c in cues if c in text)


def _words(s: Any) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$]+", _text(s)) if len(w) > 3}


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def choice_confidence(probs: list[float]) -> float:
    n = len(probs)
    return 1.0 if n == 1 else max(0.0, (max(probs) - 1 / n) / (1 - 1 / n))


def score_confidence(probs: list[float]) -> float:
    n = len(probs)
    m = probs.index(max(probs))
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return 1.0 if even == 0 else max(0.0, 1 - spread / even)


def choice_answer(names: list[str], logits: list[float]) -> dict[str, Any]:
    probs = _softmax(logits)
    best = max(range(len(names)), key=lambda i: probs[i])
    return {
        "type": "choice",
        "choice": names[best],
        "confidence": round(choice_confidence(probs), 3),
        "probabilities": {n: round(p, 3) for n, p in zip(names, probs)},
    }


def score_answer(levels: list[Any], intensity: float, sharpness: float) -> dict[str, Any]:
    """intensity in [0, 1] is the position on the scale; sharpness controls spread."""
    n = len(levels)
    centre = intensity * (n - 1)
    probs = _softmax([-sharpness * (i - centre) ** 2 for i in range(n)])
    return {
        "type": "score",
        "score": round(sum(i * p for i, p in enumerate(probs)), 3),
        "confidence": round(score_confidence(probs), 3),
        "legend": {str(i): lv for i, lv in enumerate(levels)},
        "probabilities": {str(i): round(p, 3) for i, p in enumerate(probs)},
    }


def noul_answer(logit: float) -> dict[str, Any]:
    return {"type": "noul", "noul": round(_sigmoid(logit), 3)}


# Structured-state questions read fields the state builders compute in code.
def _sim_suspicious(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    x = -2.0
    x += 1.4 * (s.get("auth_method") == "sms_otp") + 1.0 * bool(s.get("new_device"))
    x += 1.3 * (s.get("geo_match") is False) + 1.2 * (s.get("changes_72h_band") == "2_plus")
    x += 0.8 * (s.get("event_type") == "port_out") + 0.9 * (s.get("contact_velocity_7d_band") == "high")
    x -= 1.2 * (s.get("channel") == "retail_store") + 0.6 * (s.get("tenure_band") == "5y_plus")
    x += 0.9 * _hits(_text(s.get("interaction_notes", "")), NOUL_CUES["pressure_tactics"][0]) / 2
    return noul_answer(x + rng.gauss(0, 0.7))


def _sim_ato(state, qid, q, rng):
    p = _sim_suspicious(state, qid, q, rng)["noul"]
    return score_answer(q["criteria"], min(1, max(0, p + rng.gauss(0, 0.08))), 3.0)


def _safe_action(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    a = s.get("proposed_action", {})
    t = _text(a)
    x = 2.2 - 2.5 * _hits(t, CHOICE_CUES["approval_level"]["block"]) - 1.2 * ("prod" in t)
    x -= 1.5 * (s.get("repeated_same_call_band") == "3_plus")
    return noul_answer(x + rng.gauss(0, 0.5))


def _looping(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    x = -2.0 + 3.0 * (s.get("repeated_same_call_band") == "3_plus") + 1.0 * (s.get("step_band") == "over_budget")
    return noul_answer(x + rng.gauss(0, 0.5))


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\$\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:gb|%|days|months)", text.lower()))


def _consistent(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    draft, facts = _text(s.get("draft_reply", "")), _text(s.get("facts", {}))
    unsupported = [n for n in _numbers(draft) if n.replace(" ", "") not in facts.replace(" ", "")]
    x = 2.6 - 2.8 * len(unsupported)
    return noul_answer(x + rng.gauss(0, 0.4))


def _price_promise(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    draft = _text(s.get("draft_reply", ""))
    promise = any(
        w in draft for w in ["i'll credit", "i will credit", "we'll waive", "i can refund", "i'll apply a", "free for"]
    )
    allowed = isinstance(s.get("facts"), dict) and s["facts"].get("credit_allowed")
    x = -3.0 + 5.0 * (promise and not allowed)
    return noul_answer(x + rng.gauss(0, 0.3))


def _best_offer(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    need = _words(s.get("stated_need", "")) | _words(s.get("usage", {}))
    names = list(q["criteria"])
    logits = [1.2 * len(need & _words(q["criteria"][n])) + rng.gauss(0, 0.4) for n in names]
    return choice_answer(names, logits)


def _answers(state, qid, q, rng):
    s = state if isinstance(state, dict) else {}
    ask = _words(s.get("customer_question", "")) - {"i", "do", "the", "what", "how", "on", "and", "a", "my", "is"}
    hit = len(ask & _words(s.get("draft_reply", ""))) / max(1, len(ask))
    return noul_answer(-1.5 + 5.0 * hit + rng.gauss(0, 0.4))


CUSTOM: dict[str, CustomFn] = {
    "answers_question": _answers,
    "suspicious": _sim_suspicious,
    "ato_risk": _sim_ato,
    "safe_action": _safe_action,
    "looping": _looping,
    "consistent_with_facts": _consistent,
    "price_promise": _price_promise,
    "best_offer": _best_offer,
}


def _focus(state: Any) -> Any:
    """The part of the state a question is about (a crude stand-in for attention)."""
    if isinstance(state, dict) and "latest_message" in state:
        return [state["latest_message"], state.get("recent_turns", [])[-2:]]
    return state


def answer_question(state: Any, qid: str, q: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if qid in CUSTOM:
        return CUSTOM[qid](state, qid, q, rng)
    text = _text(_focus(state))
    kind = q["type"]
    if kind == "choice":
        names = list(q["criteria"])
        cues = CHOICE_CUES.get(qid, {})
        logits = []
        for n in names:
            lex = _hits(text, cues[n]) if n in cues else 0
            overlap = len(_words(text) & _words(f"{n} {q['criteria'][n]}"))
            logits.append(2.4 * lex + 0.2 * overlap + rng.gauss(0, 0.5))
        return choice_answer(names, logits)
    if kind == "score":
        cues = SCORE_CUES.get(qid, [])
        raw = sum(w for c, w in cues if c in text) + rng.gauss(0, 0.5)
        return score_answer(q["criteria"], _sigmoid(raw - 1.0), 1.6)
    if kind == "noul":
        pos, neg = NOUL_CUES.get(qid, ([], []))
        x = -1.6 + 2.6 * _hits(text, pos) - 2.6 * _hits(text, neg) + rng.gauss(0, 0.6)
        return noul_answer(x)
    raise JevError(f"mock does not support question type {kind!r}")


class MockJev:
    backend = "mock"

    def __init__(self, model: str):
        self.model = model
        self.fail_rate = float(os.environ.get("JEV_MOCK_FAIL_RATE", "0"))
        self.slow_rate = float(os.environ.get("JEV_MOCK_SLOW_RATE", "0"))
        self.realtime = os.environ.get("JEV_MOCK_REALTIME", "0") == "1"
        self._calls = 0

    def ask(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str | None = None, timeout_s: float = 5.0
    ) -> JevResponse:
        body = build_request(state, questions, model or self.model)
        seed = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        rng = random.Random(int(seed[:16], 16))
        self._calls += 1
        tokens = max(1, len(json.dumps(body)) // 4)
        latency = min(480.0, 70 + 0.03 * tokens + rng.lognormvariate(4.2, 0.45))
        chaos = random.Random(f"{seed}:{self._calls}")
        if chaos.random() < self.slow_rate:
            latency = 900 + chaos.random() * 600
        if self.realtime:
            time.sleep(min(latency, timeout_s * 1000) / 1000)
        if latency > timeout_s * 1000:
            raise JevTimeout(f"timeout after {timeout_s * 1000:.0f} ms")
        if chaos.random() < self.fail_rate:
            raise JevError("HTTP 529: overloaded (injected by mock)")
        answers = {qid: answer_question(state, qid, q, random.Random(f"{seed}:{qid}")) for qid, q in questions.items()}
        model_id = "mock-" + (model or self.model)
        usage = {"input_tokens": tokens, "output_tokens": 0}
        raw = {"model": model_id, "answers": answers, "usage": usage}
        return JevResponse(model_id, answers, usage, round(latency, 1), self.backend, raw)
