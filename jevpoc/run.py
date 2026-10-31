"""One command per level:  python -m jevpoc.run l1 ... l8  (or `all`)."""

from __future__ import annotations

import argparse
import json
import sys

from jevpoc.client import get_client, pretty
from jevpoc.gateway.core import Gateway
from jevpoc.usecases import (
    agent_guardrail,
    answer_verify,
    care_intent,
    escalation,
    next_best_offer,
    noc_triage,
    sim_change,
)


def banner(title: str, backend: str) -> None:
    print(f"\n=== {title}  [backend: {backend}] " + "=" * max(0, 60 - len(title)))


def show(label: str, d) -> None:
    keep = {k: v for k, v in d.details.items() if k not in ("input_tokens", "cost_usd")}
    tag = " (fallback)" if d.fallback_used else ""
    print(f"- {label}\n    action: {d.action}{tag}   reason: {d.reason_code}   {d.latency_ms:.0f} ms")
    if keep:
        print(f"    {json.dumps(keep)}")


def l1() -> None:
    """Hello Jev: one ticket, one question of each type, raw request and response."""
    client = get_client()
    banner("L1 Hello Jev", client.backend)
    state = {
        "ticket": "Our API integration has been returning HTTP 500 errors since this morning. "
        "We can't process any orders. Please help ASAP."
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle `ticket`?",
            "criteria": {
                "billing": "Charges, refunds, invoices",
                "technical": "Errors, outages, bugs",
                "sales": "Pricing, quotes, plan upgrades",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the author of `ticket`?",
            "criteria": ["Calm", "Mildly annoyed", "Frustrated", "Angry"],
        },
        "is_urgent": {"type": "noul", "instructions": "`ticket` needs a response within the hour."},
    }
    print("request:\n" + pretty({"model": client.model, "state": state, "questions": questions}))
    r = client.ask(state, questions)
    print("response:\n" + pretty(r.raw))
    print(
        f"department={r.choice('department')} (conf {r.confidence('department'):.2f}), "
        f"frustration={r.score('frustration'):.2f}/3, urgent p={r.noul('is_urgent'):.2f}, "
        f"{r.latency_ms:.0f} ms, {r.input_tokens} tokens, ${r.cost_usd:.7f}"
    )


def l2(gw: Gateway) -> None:
    banner("L2 Care intent routing", gw.client.backend)
    for ex in care_intent.EXAMPLES:
        show(ex["latest_message"][:70], gw.decide(care_intent.NAME, ex))
    d = gw.decide(care_intent.NAME, care_intent.EXAMPLES[0])
    print("  masked state sent for example 1:", json.dumps(d.state)[:220])


def l3(gw: Gateway) -> None:
    banner("L3 Escalation gate (re-scored every customer turn)", gw.client.backend)
    turns = escalation.EXAMPLES[0]["turns"]
    for i, t in enumerate(turns):
        if t["role"] != "customer":
            continue
        d = gw.decide(escalation.NAME, {"turns": turns[: i + 1]})
        show(f"turn {i + 1}: {t['text'][:60]}", d)
        if d.action.startswith("handoff"):
            print("    -> handed off to a person; bot stops here")
            break


def l4(gw: Gateway) -> None:
    banner("L4 Answer verification before send", gw.client.backend)
    for ex in answer_verify.EXAMPLES:
        d = gw.decide(answer_verify.NAME, ex)
        show(f"draft: {ex['draft_reply'][:66]}", d)
        if d.action.startswith("regenerate") and ex.get("regenerated"):
            d2 = gw.decide(answer_verify.NAME, {**ex, "draft_reply": ex["regenerated"]})
            show(f"  regenerated once: {ex['regenerated'][:52]}", d2)
            if d2.action != "send":
                print("    -> still failing after one retry: hold for an agent")


def l5(gw: Gateway) -> None:
    banner("L5 SIM-change / port-out risk gate", gw.client.backend)
    for ex in sim_change.EXAMPLES:
        d = gw.decide(sim_change.NAME, ex)
        show(f"{ex['event_type']} via {ex['channel']} / {ex['auth_method']}", d)
    print("  state sent for example 3 (no MSISDN, account id or name):", json.dumps(d.state)[:200])


def l6(gw: Gateway) -> None:
    banner("L6a NOC alarm triage", gw.client.backend)
    for ex in noc_triage.EXAMPLES:
        show(f"{ex['incident_id']}: {ex['alarms'][-1]['text'][:55]}", gw.decide(noc_triage.NAME, ex))
    banner("L6b Next-best-offer", gw.client.backend)
    for ex in next_best_offer.EXAMPLES:
        show(ex["stated_need"][:70], gw.decide(next_best_offer.NAME, ex))
    banner("L6c Agent guardrail", gw.client.backend)
    for ex in agent_guardrail.EXAMPLES:
        a = ex["proposed_action"]
        show(f"{a['tool']}({json.dumps(a['args'])}) in {a['environment']}", gw.decide(agent_guardrail.NAME, ex))


def l7(gw: Gateway) -> None:
    banner("L7 Gateway: timeout, breaker, fallback, log", gw.client.backend)
    ex = sim_change.EXAMPLES[1]
    print("Normal call:")
    show("esim_transfer", gw.decide(sim_change.NAME, ex))
    print("Timeout (budget 1 ms) -> rules fallback, fails toward more review:")
    show("esim_transfer", gw.decide(sim_change.NAME, ex, timeout_ms=1))
    print("Five timeouts open the circuit; next call skips Jev:")
    for _ in range(4):
        gw.decide(sim_change.NAME, ex, timeout_ms=1)
    print(f"  breaker: {gw.breaker(sim_change.NAME).state}")
    show("esim_transfer", gw.decide(sim_change.NAME, ex))
    print("Decision log stats:")
    for s in gw.log.stats():
        print(f"  {s}")
    print("Start the HTTP gateway and dashboard with:  uvicorn jevpoc.gateway.app:app --port 8000")


def l8() -> None:
    from jevpoc.eval.harness import main as eval_main

    eval_main([])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("level", choices=[f"l{i}" for i in range(1, 9)] + ["all"])
    args = p.parse_args(argv)
    levels = [f"l{i}" for i in range(1, 9)] if args.level == "all" else [args.level]
    gw = None
    for lv in levels:
        if lv in ("l1", "l8"):
            {"l1": l1, "l8": l8}[lv]()
            continue
        gw = gw or Gateway()
        globals()[lv](gw)


if __name__ == "__main__":
    sys.exit(main())
