"""L5 SIM-change / port-out risk gate: hard rules first, Jev judges the pattern, code picks the action."""

from __future__ import annotations

from typing import Any

from jevpoc.masking import mask
from jevpoc.usecases.base import band

NAME = "fraud.sim_change"


def features(ctx: dict[str, Any]) -> dict[str, Any]:
    """All counting, distance and date maths happens here, not in Jev."""
    return {
        "event_type": ctx["event_type"],
        "channel": ctx["channel"],
        "auth_method": ctx["auth_method"],
        "new_device": ctx.get("device_age_days", 999) < 2,
        "tenure_band": band(ctx.get("tenure_months", 0), [3, 24, 60], ["under_3m", "3m_2y", "2_5y", "5y_plus"]),
        "changes_72h_band": band(ctx.get("changes_72h", 0), [1, 2], ["none", "1", "2_plus"]),
        "geo_match": ctx.get("geo_distance_km", 0) < 50,
        "contact_velocity_7d_band": band(ctx.get("contacts_7d", 0), [2, 5], ["low", "medium", "high"]),
    }


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    s = features(ctx)
    s["interaction_notes"] = mask(ctx.get("interaction_notes", ""), [ctx.get("customer_name", "")])
    return s


def hard_rules(ctx: dict[str, Any]):
    if ctx.get("port_freeze") and ctx["event_type"] == "port_out":
        return "hold", "number_lock_on"
    if ctx.get("fraud_flag"):
        return "hold", "account_fraud_flag"
    if ctx["auth_method"] == "none":
        return "hold", "no_authentication"
    return None


def questions(entry, ctx):
    return entry.question_payload()


def policy(resp, ctx, th):
    p = resp.noul("suspicious")
    ato = resp.score("ato_risk")
    details = {
        "suspicious": round(p, 3),
        "ato_risk": round(ato, 2),
        "pressure": round(resp.noul("pressure_tactics"), 3),
    }
    rec = resp.choice("recommended_action") if "recommended_action" in resp.answers else None
    details["recommended_action"] = rec
    raised = rec is not None and resp.confidence("recommended_action") >= th["action_confidence"]
    if p >= th["hold"] or ato >= th["ato_hold"] or (raised and rec in ("hold", "analyst")):
        details["notify"] = "fraud_analyst"
        return (
            "hold",
            "high_takeover_risk" if p >= th["hold"] or ato >= th["ato_hold"] else f"jev_recommends_{rec}",
            details,
        )
    if raised and rec == "step_up":
        return "step_up", "jev_recommends_step_up", details
    if p >= th["step_up"]:
        return "step_up", "medium_takeover_risk", details
    return "allow", "low_takeover_risk", details


def existing_rules(ctx: dict[str, Any]) -> tuple[str, str]:
    """Today's static rule set (fallback and L8 baseline)."""
    f = features(ctx)
    if f["event_type"] == "port_out" and f["changes_72h_band"] == "2_plus":
        return "hold", "rule:port_out_after_changes"
    if f["auth_method"] == "sms_otp" and f["new_device"]:
        return "step_up", "rule:otp_new_device"
    return "allow", "rule:no_match"


def fallback(ctx):
    action, why = existing_rules(ctx)
    return ("step_up" if action == "allow" else action), f"{why}|fail_to_review"


EXAMPLES = [
    {
        "msisdn": "212-555-0199",
        "account_id": "482019384",
        "customer_name": "R. Gomez",
        "event_type": "sim_swap",
        "channel": "retail_store",
        "auth_method": "id_scan",
        "device_age_days": 0,
        "tenure_months": 96,
        "changes_72h": 0,
        "geo_distance_km": 3,
        "contacts_7d": 1,
        "interaction_notes": "Customer upgraded phone in store, ID verified.",
    },
    {
        "msisdn": "646-555-0110",
        "account_id": "771203946",
        "customer_name": "K. Shah",
        "event_type": "esim_transfer",
        "channel": "web",
        "auth_method": "sms_otp",
        "device_age_days": 0,
        "tenure_months": 30,
        "changes_72h": 1,
        "geo_distance_km": 12,
        "contacts_7d": 2,
        "interaction_notes": "Online eSIM transfer to a new phone.",
    },
    {
        "msisdn": "718-555-0175",
        "account_id": "309118274",
        "customer_name": "T. Nguyen",
        "event_type": "port_out",
        "channel": "care_phone",
        "auth_method": "sms_otp",
        "device_age_days": 0,
        "tenure_months": 40,
        "changes_72h": 3,
        "geo_distance_km": 1400,
        "contacts_7d": 6,
        "interaction_notes": "Caller said it is urgent, lost my phone, traveling, needs it right now. Hesitated on PIN.",
    },
    {
        "msisdn": "332-555-0133",
        "account_id": "120044781",
        "customer_name": "P. Brown",
        "event_type": "port_out",
        "channel": "web",
        "auth_method": "pin",
        "port_freeze": True,
        "device_age_days": 400,
        "tenure_months": 70,
        "changes_72h": 0,
        "geo_distance_km": 5,
        "contacts_7d": 0,
    },
]
