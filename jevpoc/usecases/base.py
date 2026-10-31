"""Shared helpers for state builders and policies."""

from __future__ import annotations

import json
from typing import Any

from jevpoc.client import JevResponse

Action = tuple[str, str]  # (action, reason_code)


def band(value: float, edges: list[float], labels: list[str]) -> str:
    """Bucket a number in code so Jev never has to compare numbers."""
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def fit_turns(turns: list[Any], budget_tokens: int, fixed: Any = None) -> list[Any]:
    """Keep the most recent turns that fit the state budget (oldest dropped first)."""
    used = len(json.dumps(fixed or {})) // 4
    kept: list[Any] = []
    for t in reversed(turns):
        cost = len(json.dumps(t)) // 4 + 1
        if used + cost > budget_tokens:
            break
        kept.append(t)
        used += cost
    return list(reversed(kept))


def top2(resp: JevResponse, qid: str) -> list[str]:
    probs = resp.answers[qid].get("probabilities", {})
    return sorted(probs, key=probs.get, reverse=True)[:2]
