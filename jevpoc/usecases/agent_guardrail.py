"""L6c Internal AI-agent guardrail: before each tool call, Jev judges safety and progress; code gates."""

from __future__ import annotations

import json
from typing import Any

from jevpoc.masking import mask
from jevpoc.usecases.base import band

NAME = "platform.agent_guardrail"

BLOCKED_TOOLS = {"delete_customer", "issue_refund", "drop_table", "disable_monitoring"}
DESTRUCTIVE = ("delete", "drop", "truncate", "destroy", "wipe", "purge", "terminate", "revoke", "rm -")
WRITE_TOOLS = {"update_record", "restart_service", "scale_service", "apply_config"}


def _sig(step: dict[str, Any]) -> str:
    return json.dumps([step.get("tool"), step.get("args")], sort_keys=True)


def build_state(ctx: dict[str, Any], entry) -> dict[str, Any]:
    action = ctx["proposed_action"]
    recent = ctx.get("recent_steps", [])[-5:]
    repeats = sum(1 for s in recent if _sig(s) == _sig(action))
    return mask(
        {
            "goal": ctx["goal"],
            "proposed_action": action,
            "environment": action.get("environment", "test"),
            "recent_steps": [{"tool": s["tool"], "result": s.get("result", "")} for s in recent],
            "repeated_same_call_band": band(repeats, [1, 3], ["0", "1_2", "3_plus"]),
            "step_band": "over_budget"
            if ctx.get("step_count", 0) > ctx.get("step_budget", 30) * 0.8
            else "within_budget",
            "policy_excerpt": ctx.get(
                "policy_excerpt", "Production writes need a change ticket. Agents never move money."
            ),
        }
    )


def hard_rules(ctx):
    tool = ctx["proposed_action"]["tool"]
    if tool in BLOCKED_TOOLS:
        return "block", f"tool_denylisted:{tool}"
    call = (tool + " " + json.dumps(ctx["proposed_action"].get("args", {}))).lower()
    if any(v in call for v in DESTRUCTIVE):
        return "require_approval:owner", "destructive_action"
    if ctx.get("step_count", 0) >= ctx.get("step_budget", 30):
        return "stop", "step_budget_exhausted"
    return None


def questions(entry, ctx):
    return entry.question_payload()


def policy(resp, ctx, th):
    a = ctx["proposed_action"]
    details = {
        "safe": round(resp.noul("safe_action"), 3),
        "on_goal": round(resp.noul("on_goal"), 3),
        "looping": round(resp.noul("looping"), 3),
        "approval_level": resp.choice("approval_level"),
    }
    if details["looping"] >= th["loop_stop"]:
        return "stop", "agent_looping", details
    if details["approval_level"] == "block":
        return "block", "jev_says_block", details
    prod_write = a.get("environment") == "prod" and a["tool"] in WRITE_TOOLS
    if prod_write or details["safe"] < th["safe_min"] or details["approval_level"] in ("peer", "owner"):
        return f"require_approval:{details['approval_level']}", "needs_human_check", details
    if details["on_goal"] < th["on_goal_min"]:
        return "require_approval:off_goal", "off_goal", details
    return "allow", "safe_and_on_goal", details


def fallback(ctx):
    return "require_approval:fallback", "guardrail_unavailable"


EXAMPLES = [
    {
        "goal": "Find why order 5512 is delayed",
        "step_count": 3,
        "proposed_action": {"tool": "query_orders", "args": {"order": "5512"}, "environment": "test"},
        "recent_steps": [{"tool": "list_tables", "args": {}, "result": "ok"}],
    },
    {
        "goal": "Fix the failing billing job",
        "step_count": 6,
        "proposed_action": {"tool": "restart_service", "args": {"svc": "billing-job"}, "environment": "prod"},
    },
    {
        "goal": "Fix the failing billing job",
        "step_count": 12,
        "proposed_action": {"tool": "get_logs", "args": {"svc": "billing-job"}, "environment": "prod"},
        "recent_steps": [{"tool": "get_logs", "args": {"svc": "billing-job"}, "result": "timeout, retry"}] * 4,
    },
    {
        "goal": "Resolve the customer complaint",
        "step_count": 4,
        "proposed_action": {"tool": "issue_refund", "args": {"amount": 250}, "environment": "prod"},
    },
]
