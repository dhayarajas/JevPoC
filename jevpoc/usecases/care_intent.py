"""L2 Care intent routing: Jev picks the intent; code decides route / clarify / agent."""

from __future__ import annotations

from typing import Any

from jevpoc.masking import mask
from jevpoc.usecases.base import band, fit_turns, top2

NAME = "care.intent"

QUEUES = {
    "billing_question": "billing",
    "payment_arrangement": "billing_collections",
    "plan_change": "sales_care",
    "device_tech_support": "device_support",
    "network_coverage": "network_support",
    "activation_sim": "activations",
    "order_status": "orders",
    "international_roaming": "travel_support",
    "cancel_service": "retention",
    "fraud_unrecognised": "fraud",
    "device_upgrade": "sales_care",
}

# Today's IVR-style keyword router: first match wins. Used as fallback and as the L8 baseline arm.
KEYWORDS = [
    ("cancel_service", ["cancel", "close my account"]),
    ("fraud_unrecognised", ["fraud", "hacked"]),
    ("payment_arrangement", ["payment arrangement", "extension"]),
    ("billing_question", ["bill", "charge"]),
    ("activation_sim", ["activate", "sim"]),
    ("order_status", ["order", "tracking"]),
    ("international_roaming", ["international", "roaming"]),
    ("device_upgrade", ["upgrade"]),
    ("plan_change", ["plan"]),
    ("network_coverage", ["signal", "coverage"]),
    ("device_tech_support", ["phone", "screen"]),
]


def keyword_intent(text: str) -> str | None:
    t = text.lower()
    for intent, words in KEYWORDS:
        if any(w in t for w in words):
            return intent
    return None


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    names = [ctx.get("customer_name", "")]
    acct = ctx.get("account", {})
    fixed = {
        "channel": ctx.get("channel", "chat"),
        "latest_message": ctx["latest_message"],
        "account_summary": {
            "plan": acct.get("plan"),
            "lines": acct.get("lines"),
            "tenure_band": band(acct.get("tenure_months", 0), [12, 60], ["under_1y", "1_5y", "5y_plus"]),
            "open_tickets": acct.get("open_tickets", 0),
        },
    }
    fixed = mask(fixed, names)
    turns = mask(ctx.get("recent_turns", [])[-6:], names)
    fixed["recent_turns"] = fit_turns(turns, entry.state_budget_tokens, fixed)
    return fixed


def questions(entry, ctx):
    return entry.question_payload()


def policy(resp, ctx, th):
    intent = resp.choice("intent")
    conf = resp.confidence("intent")
    details = {"intent": intent, "confidence": round(conf, 3), "queue": QUEUES[intent]}
    if resp.noul("needs_human") >= th["needs_human"]:
        return "agent", "customer_asked_for_human", details
    if resp.noul("in_scope") < th["in_scope_min"]:
        return "out_of_scope", "not_an_account_request", details
    if conf >= th["route"]:
        return f"route:{QUEUES[intent]}", f"intent={intent}", details
    if conf >= th["clarify"]:
        a, b = top2(resp, "intent")
        details["clarify_between"] = [a, b]
        return f"clarify:{a}|{b}", "medium_confidence", details
    return "agent", "low_confidence", details


def fallback(ctx):
    intent = keyword_intent(ctx["latest_message"])
    return (f"route:{QUEUES[intent]}", f"keyword={intent}") if intent else ("agent", "no_keyword")


EXAMPLES = [
    {
        "customer_name": "Maria Lopez",
        "msisdn": "917-555-0142",
        "latest_message": "My bill went up by $35 this month and I don't know why. Maria, 917-555-0142",
        "account": {"plan": "Unlimited Plus", "lines": 3, "tenure_months": 74, "open_tickets": 0},
    },
    {
        "customer_name": "Dev Patel",
        "latest_message": "I'm flying to Lisbon on Friday, will my phone work over there?",
        "account": {"plan": "Unlimited Welcome", "lines": 1, "tenure_months": 8},
    },
    {
        "customer_name": "Sam Lee",
        "latest_message": "Calls keep dropping at home and I'm getting no signal in the kitchen.",
        "recent_turns": [{"role": "customer", "text": "Hi"}, {"role": "bot", "text": "How can I help?"}],
        "account": {"plan": "Unlimited Ultimate", "lines": 2, "tenure_months": 30},
    },
    {
        "customer_name": "Ana Ruiz",
        "latest_message": "There's a new line on my account I don't recognize, someone else added it.",
        "account": {"plan": "Unlimited Plus", "lines": 4, "tenure_months": 120},
    },
    {
        "customer_name": "Joe Kim",
        "latest_message": "Just let me talk to a real person please.",
        "account": {"plan": "Unlimited Welcome", "lines": 1, "tenure_months": 3},
    },
    {
        "customer_name": "Li Wei",
        "latest_message": "hey",
        "account": {"plan": "Unlimited Welcome", "lines": 1, "tenure_months": 14},
    },
]
