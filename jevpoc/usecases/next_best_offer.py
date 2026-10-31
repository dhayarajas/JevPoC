"""L6b Next-best-offer: eligibility and price in code, Jev ranks eligible offers by fit to the stated need."""

from __future__ import annotations

import copy
from typing import Any

from jevpoc.masking import mask

NAME = "sales.next_best_offer"

OFFERS = {
    "travelpass": {
        "desc": "TravelPass for a trip abroad: use your phone and plan while travelling internationally, by the day",
        "price": "$12/day",
        "segments": {"consumer"},
    },
    "plus_upgrade": {
        "desc": "Move to Unlimited Plus: more premium data and 30 GB hotspot for streaming and work",
        "price": "+$10/line",
        "segments": {"consumer"},
    },
    "add_line": {
        "desc": "Add a line for a family member, kid or tablet at a multi-line discount",
        "price": "$30/mo",
        "segments": {"consumer"},
    },
    "home_internet": {
        "desc": "5G Home Internet bundle discount for home wifi and streaming",
        "price": "$35/mo",
        "segments": {"consumer"},
    },
    "device_protect": {
        "desc": "Device protection for a cracked screen, broken or lost phone",
        "price": "$17/mo",
        "segments": {"consumer", "business"},
    },
}
SEGMENT_DEFAULT = {"consumer": "plus_upgrade", "business": "device_protect"}


def eligible(ctx: dict[str, Any]) -> list[str]:
    owned = set(ctx.get("owned", []))
    return [o for o, v in OFFERS.items() if ctx.get("segment", "consumer") in v["segments"] and o not in owned]


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    return mask(
        {
            "stated_need": ctx["stated_need"],
            "usage": {k: v for k, v in ctx.get("usage", {}).items() if k not in FAIRNESS_EXCLUDED},
            "eligible_offers": eligible(ctx)[:8],
            "plan": ctx.get("plan"),
            "recent_turns": ctx.get("recent_turns", [])[-4:],
        },
        [ctx.get("customer_name", "")],
    )


def questions(entry, ctx):
    q = entry.question_payload()
    q["best_offer"] = copy.deepcopy(q["best_offer"])
    q["best_offer"]["criteria"] = {o: OFFERS[o]["desc"] for o in eligible(ctx)}
    return q


# Never sent to Jev and never used to choose an offer.
FAIRNESS_EXCLUDED = {"age", "gender", "ethnicity", "zip", "credit_score", "disability"}
MAX_OFFERS_30D = 2


def hard_rules(ctx):
    if ctx.get("marketing_opt_out"):
        return "no_offer", "marketing_opt_out"
    if ctx.get("offers_presented_30d", 0) >= MAX_OFFERS_30D:
        return "no_offer", "contact_cap_reached"
    if not eligible(ctx):
        return "no_offer", "nothing_eligible"
    return None


def policy(resp, ctx, th):
    best = resp.choice("best_offer")
    details = {
        "best_offer": best,
        "confidence": round(resp.confidence("best_offer"), 3),
        "open_to_offer": round(resp.noul("open_to_offer"), 3),
        "save_risk": round(resp.score("save_risk"), 2),
    }
    if details["save_risk"] >= th["save_desk"]:
        return "route:save_desk", "retention_first", details
    if details["open_to_offer"] < th["open_min"]:
        return "no_offer", "not_receptive", details
    if details["confidence"] < th["offer_confidence"]:
        return "no_offer", "unsure_which_offer", details
    details["price"] = OFFERS[best]["price"]  # price comes from the pricing table, never from Jev
    return f"offer:{best}", "fit_to_need", details


def fallback(ctx):
    return "no_offer", "no_decision_no_offer"


EXAMPLES = [
    {
        "customer_name": "A. Diaz",
        "segment": "consumer",
        "plan": "Unlimited Welcome",
        "stated_need": "I'm going to Italy next month for a trip, what are my options for using my phone abroad?",
        "usage": {"data_gb_avg": 18, "intl_days_last_year": 0},
    },
    {
        "customer_name": "M. Chen",
        "segment": "consumer",
        "plan": "Unlimited Welcome",
        "stated_need": "I keep running out of hotspot when I work from the train, looking for a better deal for streaming and work.",
        "usage": {"data_gb_avg": 60},
    },
    {
        "customer_name": "J. Olsen",
        "segment": "consumer",
        "plan": "Unlimited Plus",
        "stated_need": "The bill is too expensive and T-Mobile is cheaper, I want to cancel.",
    },
    {
        "customer_name": "R. Ito",
        "segment": "consumer",
        "plan": "Unlimited Plus",
        "marketing_opt_out": True,
        "stated_need": "Just checking my data usage.",
    },
]
