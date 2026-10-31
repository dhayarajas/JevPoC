"""L4 Answer verification: an LLM drafts, Jev checks the draft against KB facts, code gates the send."""

from __future__ import annotations

import re
from typing import Any

from jevpoc.masking import mask
from jevpoc.usecases.base import looks_like_instruction

NAME = "care.answer_verify"

PROMISE = re.compile(r"(?i)\b(i'?ll|i will|we'?ll|we will|i can)\s+(credit|refund|waive|apply a|give you)")
MONEY = re.compile(r"\$\s?\d+(?:\.\d{2})?")


def unsupported_amounts(draft: str, facts: dict[str, Any]) -> list[str]:
    """Deterministic check in code: every dollar amount in the draft must appear in facts."""
    fact_text = " ".join(str(v) for v in facts.values()).replace(" ", "")
    return [m for m in MONEY.findall(draft) if m.replace(" ", "") not in fact_text]


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    return mask(
        {
            "customer_question": ctx["customer_question"],
            "draft_reply": ctx["draft_reply"],
            "facts": ctx["facts"],
        },
        [ctx.get("customer_name", "")],
    )


def hard_rules(ctx: dict[str, Any]):
    if looks_like_instruction(ctx["draft_reply"]):
        return "hold_for_agent", "instruction_like_text_in_draft"
    return None


def questions(entry, ctx):
    return entry.question_payload()


def policy(resp, ctx, th):
    amounts = unsupported_amounts(ctx["draft_reply"], ctx["facts"])
    details = {
        "consistent": round(resp.noul("consistent_with_facts"), 3),
        "price_promise": round(resp.noul("price_promise"), 3),
        "answers_question": round(resp.noul("answers_question"), 3),
        "failure_type": resp.choice("failure_type"),
        "unsupported_amounts": amounts,
    }
    promise_in_code = bool(PROMISE.search(ctx["draft_reply"])) and not ctx["facts"].get("credit_allowed")
    if promise_in_code or details["price_promise"] >= th["promise_block"]:
        return "block:price_promise", "money_promise_needs_agent", details
    if amounts or details["consistent"] < th["consistent_min"]:
        return f"regenerate:{details['failure_type']}", "draft_not_supported_by_facts", details
    if details["answers_question"] < th["answers_min"]:
        return "regenerate:off_topic", "draft_does_not_answer", details
    return "send", "verified", details


def fallback(ctx):
    return "hold_for_agent", "verifier_unavailable"


FACTS = {
    "plan": "Unlimited Plus",
    "monthly_price": "$80 per line with Auto Pay",
    "premium_hotspot": "30 GB",
    "upgrade_fee": "$35 per device",
    "return_window": "30 days",
    "credit_allowed": False,
}

EXAMPLES = [
    {
        "customer_question": "How much hotspot do I get on Unlimited Plus and what does it cost?",
        "draft_reply": "Unlimited Plus is $80 per line with Auto Pay and includes 30 GB of premium hotspot.",
        "facts": FACTS,
    },
    {
        "customer_question": "How much hotspot do I get on Unlimited Plus and what does it cost?",
        "draft_reply": "Unlimited Plus is $65 per line and includes 50 GB of premium hotspot.",
        "facts": FACTS,
        "regenerated": "Unlimited Plus is $80 per line with Auto Pay and includes 30 GB of premium hotspot.",
    },
    {
        "customer_question": "Why was I charged an upgrade fee?",
        "draft_reply": "Sorry about that! I'll credit the $35 upgrade fee back to your account today.",
        "facts": FACTS,
    },
]
