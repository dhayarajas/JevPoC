"""L6a NOC alarm and ticket triage: Jev rates severity and picks a fault domain; code routes and pages."""

from __future__ import annotations

import re
from typing import Any

from jevpoc.masking import token
from jevpoc.usecases.base import band

NAME = "noc.triage"

TEAMS = {
    "ran": "ran_ops",
    "transport": "transport_ops",
    "core": "core_ops",
    "power": "field_power",
    "customer_premises": "home_support",
}


def site_token(element: str) -> str:
    site, _, rest = element.partition("/")
    return token("SITE", site) + (f"/{rest}" if rest else "")


SITE_CODE = re.compile(r"\b[A-Z]{3}\d{3,5}\b")  # CLLI-style site codes such as NYC0412


def scrub_sites(obj: Any, sites: set[str]) -> Any:
    """Replace known and site-code-shaped identifiers in free text with the same site tokens."""
    if isinstance(obj, dict):
        return {k: scrub_sites(v, sites) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_sites(v, sites) for v in obj]
    if isinstance(obj, str):
        for s in sorted(sites, key=len, reverse=True):
            obj = re.sub(rf"\b{re.escape(s)}\b", token("SITE", s), obj)
        return SITE_CODE.sub(lambda m: token("SITE", m.group(0)), obj)
    return obj


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    sites = {a["element"].split("/")[0] for a in ctx["alarms"]}
    seen, alarms = set(), []
    for a in sorted(ctx["alarms"], key=lambda a: a["time"]):
        key = (a["element"], a["text"])
        if key not in seen:
            seen.add(key)
            alarms.append(
                {"time": a["time"], "element": site_token(a["element"]), "text": scrub_sites(a["text"], sites)}
            )
    k = ctx.get("kpis", {})
    return {
        "alarms": alarms[-20:],
        "duplicates_removed": len(ctx["alarms"]) - len(alarms),
        "sites_affected": len({a["element"].split("/")[0] for a in alarms}),
        "kpi_bands": {
            "drop_rate": band(k.get("drop_rate_pct", 0), [1, 3], ["normal", "elevated", "high"]),
            "availability": band(k.get("availability_pct", 100), [95, 99.5], ["down", "degraded", "normal"]),
        },
        "recent_changes": scrub_sites(ctx.get("recent_changes", []), sites),
    }


def hard_rules(ctx):
    if ctx.get("priority_services") and ctx.get("kpis", {}).get("availability_pct", 100) < 95:
        return "page:p1_bridge", "priority_services_outage"
    return None


def questions(entry, ctx):
    return entry.question_payload()


def policy(resp, ctx, th):
    sev, dom = resp.score("severity"), resp.choice("fault_domain")
    details = {
        "severity": round(sev, 2),
        "severity_conf": round(resp.confidence("severity"), 3),
        "fault_domain": dom,
        "domain_conf": round(resp.confidence("fault_domain"), 3),
        "truck_roll": round(resp.noul("truck_roll"), 3),
        "change_related": round(resp.noul("change_related"), 3),
    }
    team = TEAMS[dom] if details["domain_conf"] >= th["domain_confidence"] else "noc_l2_triage"
    flags = [
        f
        for f, on in (
            ("dispatch", details["truck_roll"] >= th["truck_roll"]),
            ("link_change", details["change_related"] >= 0.6),
        )
        if on
    ]
    suffix = ("+" + "+".join(flags)) if flags else ""
    if sev >= th["page_severity"] and details["severity_conf"] >= th["page_confidence"]:
        return f"page:{team}{suffix}", "high_severity", details
    return f"ticket:{team}{suffix}", "normal_severity", details


def fallback(ctx):
    return "ticket:noc_l2_triage", "manual_triage"


EXAMPLES = [
    {
        "incident_id": "INC-1042",
        "alarms": [
            {
                "time": "02:14",
                "element": "NYC0412/gNodeB-2",
                "text": "Site power alarm: AC mains failure, running on battery",
            },
            {"time": "02:41", "element": "NYC0412/gNodeB-2", "text": "Battery low, rectifier alarm"},
            {"time": "02:41", "element": "NYC0412/gNodeB-2", "text": "Battery low, rectifier alarm"},
            {"time": "02:55", "element": "NYC0412/gNodeB-2", "text": "Cell down, sectors 1-3 outage"},
        ],
        "kpis": {"drop_rate_pct": 4.2, "availability_pct": 91.0},
        "recent_changes": [],
    },
    {
        "incident_id": "INC-1043",
        "alarms": [
            {"time": "10:02", "element": "BOS0101/router-a", "text": "Software upgrade CR-8812 applied"},
            {"time": "10:05", "element": "BOS0101/router-a", "text": "Backhaul link down, microwave LOS"},
            {"time": "10:06", "element": "BOS0102/router-a", "text": "Backhaul link down"},
        ],
        "kpis": {"drop_rate_pct": 2.1, "availability_pct": 97.0},
        "recent_changes": ["CR-8812 software upgrade on BOS routers at 10:00"],
    },
    {
        "incident_id": "INC-1044",
        "alarms": [{"time": "14:30", "element": "CPE/ONT-55812", "text": "ONT LOS at single customer premises, minor"}],
        "kpis": {"drop_rate_pct": 0.2, "availability_pct": 99.9},
    },
]
