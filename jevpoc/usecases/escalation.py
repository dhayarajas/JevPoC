"""L3 Escalation gate: re-scored on every customer turn; code decides when to hand off."""

from __future__ import annotations

from typing import Any

from jevpoc.masking import mask
from jevpoc.usecases.base import fit_turns

NAME = "care.escalation"


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    turns = mask(ctx["turns"], [ctx.get("customer_name", "")])
    return {
        "bot_attempts": sum(1 for t in turns if t["role"] == "bot"),
        "turns": fit_turns(turns, entry.state_budget_tokens),
    }


def questions(entry, ctx):
    return entry.question_payload()


def policy(resp, ctx, th):
    fr = resp.score("frustration")
    details = {
        "frustration": round(fr, 2),
        "churn": round(resp.noul("churn_language"), 3),
        "asks_for_human": round(resp.noul("asks_for_human"), 3),
        "bot_resolvable": round(resp.noul("bot_resolvable"), 3),
    }
    if details["asks_for_human"] >= th["human"]:
        return "handoff:agent", "customer_asked_for_human", details
    if details["churn"] >= th["churn_alert"]:
        return "handoff:retention", "churn_language", details
    if fr >= th["frustration_alert"] and details["bot_resolvable"] < th["resolvable_min"]:
        return "handoff:agent", "frustrated_and_not_resolvable", details
    if fr >= th["frustration_alert"]:
        return "continue:soften_tone", "frustrated_but_resolvable", details
    return "continue", "ok", details


def fallback(ctx):
    last = ctx["turns"][-1]["text"].lower()
    if any(w in last for w in ("agent", "person", "cancel", "supervisor")):
        return "handoff:agent", "keyword"
    if len(ctx["turns"]) >= 7:
        return "handoff:agent", "long_conversation"
    return "continue", "no_keyword"


EXAMPLES = [
    {
        "customer_name": "Chris",
        "turns": [
            {"role": "customer", "text": "My internet is slow, how do I reset my router?"},
            {"role": "bot", "text": "Unplug it for 30 seconds, then plug it back in."},
            {"role": "customer", "text": "Did that, still slow."},
            {"role": "bot", "text": "Let me run a line test. It looks fine on our side."},
            {"role": "customer", "text": "This is the third time this month. It's ridiculous, nobody fixes it."},
            {"role": "bot", "text": "I'm sorry about that. Let me try another step."},
            {"role": "customer", "text": "Unacceptable. I'm going to cancel and switch to T-Mobile."},
        ],
    }
]
