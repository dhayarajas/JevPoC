"""Pilot protocol (deck slides 17, 24, 46): the same three arms on all four pilots.

    python -m jevpoc.eval.protocol        # writes reports/eval/protocol.{json,html}

Arm A  today's approach (rules or keyword router already in the code)
Arm B  constrained LLM on the same question schema (System One adapter style)
Arm C  Jev, pinned model, behind the gateway policy

Every arm sees the same frozen test split. Thresholds come from the catalogue.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from typing import Any

from jevpoc import catalogue
from jevpoc.client import JevResponse, get_client
from jevpoc.eval import data, metrics
from jevpoc.eval.harness import OUT, ask_all, reliability_plot
from jevpoc.eval.llm_arm import get_llm_client
from jevpoc.usecases import answer_verify, care_intent, escalation, sim_change

# Slide 16/17 success bar. Proposals until the review board adopts them.
TARGETS = {"ece_max": 0.05, "p95_ms_max": 300.0, "cost_ratio_max": 0.10, "automated_min": 0.70}
FALSE_ACTION_TOLERANCE = {"care_intent": 0.05, "escalation": 0.10, "answer_verify": 0.02, "sim_change": 0.05}


@dataclass
class Outcome:
    action: str
    correct: bool
    automated: bool
    key_conf: float | None = None


def _care(row, resp: JevResponse | None, th, arm: str) -> Outcome:
    ctx, gold = row["context"], row["gold_action"]
    if resp is None:
        a = care_intent.fallback(ctx)[0]
        return Outcome(a, a == gold, a.startswith("route:"))
    a = care_intent.policy(resp, ctx, th)[0]
    ok = a == gold or (a.startswith("clarify:") and row["gold_intent"] in a.removeprefix("clarify:").split("|"))
    return Outcome(a, ok, a.startswith("route:") or a == "out_of_scope", resp.confidence("intent"))


def _escalation(row, resp, th, arm) -> Outcome:
    ctx, gold = row["context"], row["gold_escalate"]
    a = escalation.fallback(ctx)[0] if resp is None else escalation.policy(resp, ctx, th)[0]
    esc = a.startswith(("handoff", "alert"))
    conf = None if resp is None else max(resp.noul("asks_for_human"), resp.noul("churn_language"))
    return Outcome(a, esc == gold, not esc, conf)


def _today_verify(ctx) -> str:
    bad = answer_verify.unsupported_amounts(ctx["draft_reply"], ctx["facts"]) or answer_verify.PROMISE.search(
        ctx["draft_reply"]
    )
    return "hold_for_agent" if bad else "send"


def _answer_verify(row, resp, th, arm) -> Outcome:
    ctx, gold = row["context"], row["gold_send"]
    a = _today_verify(ctx) if resp is None else answer_verify.policy(resp, ctx, th)[0]
    send = a == "send"
    conf = None if resp is None else resp.noul("consistent_with_facts")
    return Outcome(a, send == gold, send, conf)


def _sim(row, resp, th, arm) -> Outcome:
    ctx, gold = row["context"], row["gold_fraud"]
    hard = sim_change.hard_rules(ctx)
    if resp is None:
        a = sim_change.existing_rules(ctx)[0]
    else:
        a = hard[0] if hard else sim_change.policy(resp, ctx, th)[0]
    caught = a in ("hold", "step_up")
    return Outcome(a, caught == gold, a == "allow", None if resp is None else resp.noul("suspicious"))


PILOTS = {
    "care_intent": (care_intent, _care, "gold_action"),
    "escalation": (escalation, _escalation, "gold_escalate"),
    "answer_verify": (answer_verify, _answer_verify, "gold_send"),
    "sim_change": (sim_change, _sim, "gold_fraud"),
}


def _key_truth(name: str, row: dict, o: Outcome) -> bool:
    """What the key question's probability should predict, for ECE."""
    if name == "care_intent":
        return o.correct
    return bool(row[PILOTS[name][2]]) if name != "answer_verify" else bool(row["gold_send"])


def arm_metrics(name: str, rows: list[dict], outs: list[Outcome], scored=None) -> dict[str, Any]:
    gold = [str(r[PILOTS[name][2]]) for r in rows]
    pred_ok = [o.correct for o in outs]
    auto = [o for o in outs if o.automated]
    m: dict[str, Any] = {
        "accuracy": sum(pred_ok) / len(outs),
        "macro_f1": metrics.macro_f1(gold, [g if o.correct else "wrong" for g, o in zip(gold, outs)]),
        "automated": len(auto) / len(outs),
        "false_action_rate": sum(1 for o in auto if not o.correct) / max(1, len(auto)),
    }
    confs = [(o.key_conf, _key_truth(name, r, o)) for r, o in zip(rows, outs) if o.key_conf is not None]
    if confs:
        m["ece"] = metrics.ece([c for c, _ in confs], [t for _, t in confs])
    if scored is not None:
        lat = [s.resp.latency_ms for s in scored if s.resp]
        cost = [s.resp.cost_usd for s in scored if s.resp]
        m.update(
            p50_ms=metrics.pct(lat, 50),
            p95_ms=metrics.pct(lat, 95),
            cost_per_1k_usd=1000 * sum(cost) / max(1, len(cost)),
        )
    else:
        m.update(p50_ms=0.0, p95_ms=0.0, cost_per_1k_usd=0.0)
    return m


def run_pilot(name: str, jev, llm) -> dict[str, Any]:
    mod, decide, gold_key = PILOTS[name]
    entry = catalogue.load()[mod.NAME]
    rows = [r for r in data.load(name) if r["split"] == "test"]
    th = entry.thresholds
    sc_c = ask_all(jev, mod, entry, rows)
    sc_b = ask_all(llm, mod, entry, rows)
    outs = {
        "A_today": [decide(r, None, th, "A") for r in rows],
        "B_constrained_llm": [decide(s.row, s.resp, th, "B") for s in sc_b],
        "C_jev": [decide(s.row, s.resp, th, "C") for s in sc_c],
    }
    arms = {
        "A_today": arm_metrics(name, rows, outs["A_today"]),
        "B_constrained_llm": arm_metrics(name, rows, outs["B_constrained_llm"], sc_b),
        "C_jev": arm_metrics(name, rows, outs["C_jev"], sc_c),
    }
    c = outs["C_jev"]
    by = lambda pred: [o.correct for r, o in zip(rows, c) if pred(r)]  # noqa: E731
    sub = {
        "temporal_train": by(lambda r: r["period"] == "train"),
        "temporal_post_launch": by(lambda r: r["period"] == "post_launch"),
        "spanish": by(lambda r: r.get("lang") == "es"),
    }
    relabelled = [r for r in data.load(name) if "relabel" in r]
    agree = [r["relabel"] == r[gold_key] for r in relabelled]
    conf = [(o.key_conf, _key_truth(name, r, o)) for r, o in zip(rows, c) if o.key_conf is not None]
    return {
        "use_case": mod.NAME,
        "catalogue": entry.key,
        "n_test": len(rows),
        "arms": arms,
        "bootstrap_C_vs_A": metrics.paired_bootstrap([o.correct for o in c], [o.correct for o in outs["A_today"]]),
        "bootstrap_C_vs_B": metrics.paired_bootstrap(
            [o.correct for o in c], [o.correct for o in outs["B_constrained_llm"]]
        ),
        "subsets": {k: {"n": len(v), "accuracy": sum(v) / len(v) if v else None} for k, v in sub.items()},
        "label_noise": {"n_relabelled": len(agree), "agreement": sum(agree) / max(1, len(agree))},
        "reliability": metrics.reliability([a for a, _ in conf], [b for _, b in conf]),
        "coverage_curve": metrics.coverage_curve(
            [a for a, _ in conf], [o.correct for o in c if o.key_conf is not None]
        ),
    }


def scorecard(res: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def mark(ok: bool) -> str:
        return "green" if ok else "red"

    rows: dict[str, dict[str, str]] = {}
    for name, r in res.items():
        a, b, c = r["arms"]["A_today"], r["arms"]["B_constrained_llm"], r["arms"]["C_jev"]
        rows[name] = {
            "accuracy_vs_today": mark(c["accuracy"] >= a["accuracy"]),
            "calibration_ece": mark(c.get("ece", 1) < TARGETS["ece_max"]),
            "p95_latency": mark(c["p95_ms"] < TARGETS["p95_ms_max"]),
            "cost_vs_llm": mark(c["cost_per_1k_usd"] < TARGETS["cost_ratio_max"] * max(b["cost_per_1k_usd"], 1e-12)),
            "automated_share": mark(c["automated"] >= TARGETS["automated_min"]),
            "false_action_rate": mark(c["false_action_rate"] <= FALSE_ACTION_TOLERANCE[name]),
            "fallback_failover": "green",  # covered by tests/test_gateway.py and the dashboard chaos test
            "privacy_contract": "red",  # DPA, CPNI and ZDR terms are not signed; outside this POC
        }
    tech = [n for n, r in rows.items() if all(v == "green" for k, v in r.items() if k != "privacy_contract")]
    if any(r["accuracy_vs_today"] == "red" for r in rows.values()):
        verdict = "stop: accuracy below baseline on at least one pilot"
    elif len(tech) >= 2:
        verdict = "technically green on >= 2 pilots; go is blocked until privacy and contract terms are signed"
    else:
        verdict = "extend: amber or red on latency, calibration or automation"
    return {"targets": TARGETS, "false_action_tolerance": FALSE_ACTION_TOLERANCE, "rows": rows, "verdict": verdict}


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))


def render(summary: dict[str, Any]) -> str:
    mock = "mock" in (summary["backend_c"], summary["backend_b"])
    parts = []
    if mock:
        parts.append(
            "<p class='warn'>Offline mock run. Arms B and C use stand-ins, so these numbers test the protocol, "
            "not Jev or any LLM.</p>"
        )
    sc = summary["scorecard"]
    crit = list(next(iter(sc["rows"].values())))
    head = "".join(f"<th>{n}</th>" for n in sc["rows"])
    body = "".join(
        f"<tr><td>{c}</td>"
        + "".join(f"<td class='{sc['rows'][n][c]}'>{sc['rows'][n][c]}</td>" for n in sc["rows"])
        + "</tr>"
        for c in crit
    )
    parts.append(f"<h2>Week-12 scorecard (slide 17)</h2><table><tr><th>criterion</th>{head}</tr>{body}</table>")
    parts.append(f"<p><b>Verdict:</b> {html.escape(sc['verdict'])}</p>")
    for name, r in summary["pilots"].items():
        cols = sorted({k for m in r["arms"].values() for k in m})
        t = "".join(
            f"<tr><td>{arm}</td>" + "".join(f"<td>{_fmt(m.get(k, '-'))}</td>" for k in cols) + "</tr>"
            for arm, m in r["arms"].items()
        )
        bs = r["bootstrap_C_vs_A"]
        parts.append(
            f"<h2>{html.escape(r['use_case'])} <span class='muted'>{r['catalogue']} · n_test {r['n_test']}</span></h2>"
            f"<table><tr><th>arm</th>{''.join(f'<th>{k}</th>' for k in cols)}</tr>{t}</table>"
            f"<p>C vs A paired bootstrap: diff {bs['diff']:+.3f} (95% CI {bs['ci_low']:+.3f} to {bs['ci_high']:+.3f}, "
            f"p {bs['p_value']:.3f}). Subsets: "
            + ", ".join(f"{k} {_fmt(v['accuracy'])} (n {v['n']})" for k, v in r["subsets"].items())
            + f". Label agreement on {r['label_noise']['n_relabelled']} blind relabels (simulated): "
            f"{r['label_noise']['agreement']:.3f}.</p>"
            f"<img src='data:image/png;base64,{reliability_plot(r['reliability'], name, 'key question probability')}'>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Jev pilot protocol</title><style>"
        "body{font:14px system-ui;margin:2em;max-width:1100px}table{border-collapse:collapse;margin:.5em 0}"
        "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}td:first-child{text-align:left}"
        ".green{background:#d9f2d9}.red{background:#f6d5d5}.warn{background:#fff3cd;padding:8px}.muted{color:#777;font-size:12px}"
        "img{max-width:420px}</style></head><body><h1>Jev pilot protocol: three arms, four pilots</h1>"
        f"<p class='muted'>A today · B {html.escape(summary['backend_b'])} · C {html.escape(summary['backend_c'])} "
        f"{html.escape(summary['model'])} · frozen synthetic test split</p>" + "".join(parts) + "</body></html>"
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    jev, llm = get_client(), get_llm_client()
    pilots = {n: run_pilot(n, jev, llm) for n in PILOTS}
    summary = {
        "backend_c": jev.backend,
        "backend_b": llm.backend,
        "model": jev.model,
        "pilots": pilots,
        "scorecard": scorecard(pilots),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(summary, indent=2))
    (OUT / "protocol.html").write_text(render(summary))
    print(f"\n=== Pilot protocol  [C: {jev.backend}, B: {llm.backend}] ===")
    for n, r in pilots.items():
        for arm, m in r["arms"].items():
            print(
                f"  {n:14s} {arm:18s} acc={m['accuracy']:.3f} auto={m['automated']:.3f} ece={m.get('ece', float('nan')):.3f}"
            )
    print(f"verdict: {summary['scorecard']['verdict']}")
    print(f"report: {OUT / 'protocol.html'}")
    return summary


if __name__ == "__main__":
    os.environ.setdefault("JEV_MOCK_REALTIME", "0")
    main()
