"""L8 Evaluation harness (deck slide 46): three arms on a frozen labelled set.

    python -m jevpoc.eval.harness            # writes reports/eval/report.html and summary.json

Arms: today's rules (baseline) | Jev alone | Jev + deterministic rules (the gateway policy).
Thresholds are chosen on the dev split only, then reported once on the test split.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jevpoc import catalogue  # noqa: E402
from jevpoc.client import JevError, JevResponse, get_client  # noqa: E402
from jevpoc.eval import data, metrics  # noqa: E402
from jevpoc.usecases import care_intent, sim_change  # noqa: E402

OUT = catalogue.ROOT / "reports" / "eval"
TARGET_ROUTE_PRECISION = 0.95
TARGET_FRAUD_RECALL = 0.95


@dataclass
class Scored:
    row: dict[str, Any]
    resp: JevResponse | None
    error: str | None = None


def ask_all(client, mod, entry, rows, workers: int = 8) -> list[Scored]:
    def one(r):
        try:
            return Scored(
                r,
                client.ask(
                    mod.build_state(r["context"], entry),
                    mod.questions(entry, r["context"]),
                    entry.model,
                    timeout_s=max(entry.timeout_ms / 1000, 2.0),
                ),
            )
        except JevError as e:
            return Scored(r, None, str(e))

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def ops(scored: list[Scored]) -> dict[str, float]:
    lat = [s.resp.latency_ms for s in scored if s.resp]
    cost = [s.resp.cost_usd for s in scored if s.resp]
    return {
        "p50_ms": metrics.pct(lat, 50),
        "p95_ms": metrics.pct(lat, 95),
        "cost_per_1k_usd": 1000 * sum(cost) / max(1, len(cost)),
        "errors": sum(1 for s in scored if s.error),
    }


# ---------------------------------------------------------------- care intent
def intent_actions(scored: list[Scored], th: dict[str, float]) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {"baseline_rules": [], "jev_only": [], "jev_plus_rules": []}
    for s in scored:
        ctx = s.row["context"]
        arms["baseline_rules"].append(care_intent.fallback(ctx)[0])
        if s.resp is None:
            arms["jev_only"].append("error")
            arms["jev_plus_rules"].append(care_intent.fallback(ctx)[0])
            continue
        arms["jev_only"].append("route:" + care_intent.QUEUES[s.resp.choice("intent")])
        arms["jev_plus_rules"].append(care_intent.policy(s.resp, ctx, th)[0])
    return arms


def routing_metrics(gold: list[str], acts: list[str]) -> dict[str, float]:
    n = len(gold)
    routed = [(g, a) for g, a in zip(gold, acts) if a.startswith("route:")]
    mis = sum(1 for g, a in routed if g != a)
    return {
        "action_accuracy": metrics.accuracy(gold, acts),
        "automation_rate": len(routed) / n,
        "misroute_rate": mis / n,
        "route_precision": 1 - mis / len(routed) if routed else float("nan"),
    }


def choose_route_threshold(dev: list[Scored], base_th: dict[str, float]) -> tuple[float, list[dict]]:
    sweep = []
    for t in [x / 100 for x in range(30, 100, 5)]:
        th = {**base_th, "route": t, "clarify": min(base_th["clarify"], t)}
        m = routing_metrics([s.row["gold_action"] for s in dev], intent_actions(dev, th)["jev_plus_rules"])
        sweep.append({"threshold": t, **m})
    ok = [r for r in sweep if r["route_precision"] >= TARGET_ROUTE_PRECISION]
    best = min(ok, key=lambda r: r["threshold"]) if ok else max(sweep, key=lambda r: r["route_precision"])
    return best["threshold"], sweep


def eval_care_intent(client) -> dict[str, Any]:
    entry = catalogue.load()[care_intent.NAME]
    scored = ask_all(client, care_intent, entry, data.load("care_intent"))
    dev = [s for s in scored if s.row["split"] == "dev"]
    test = [s for s in scored if s.row["split"] == "test"]
    t, sweep = choose_route_threshold(dev, entry.thresholds)
    th = {**entry.thresholds, "route": t, "clarify": min(entry.thresholds["clarify"], t)}
    gold = [s.row["gold_action"] for s in test]
    arms = {k: routing_metrics(gold, v) for k, v in intent_actions(test, th).items()}

    with_intent = [s for s in test if s.row["gold_intent"] and s.resp]
    y = [s.row["gold_intent"] for s in with_intent]
    arms["baseline_rules"]["intent_macro_f1"] = metrics.macro_f1(
        y, [care_intent.keyword_intent(s.row["context"]["latest_message"]) for s in with_intent]
    )
    jev_pred = [s.resp.choice("intent") for s in with_intent]
    arms["jev_only"]["intent_macro_f1"] = arms["jev_plus_rules"]["intent_macro_f1"] = metrics.macro_f1(y, jev_pred)
    conf = [s.resp.confidence("intent") for s in with_intent]
    correct = [p == g for p, g in zip(jev_pred, y)]
    return {
        "use_case": care_intent.NAME,
        "n_dev": len(dev),
        "n_test": len(test),
        "chosen": {
            "route_threshold": t,
            "rule": f"lowest threshold with dev route precision >= {TARGET_ROUTE_PRECISION}",
            "catalogue_placeholder": entry.thresholds["route"],
        },
        "arms": arms,
        "calibration": {"ece": metrics.ece(conf, correct), "bins": metrics.reliability(conf, correct)},
        "sweep_dev": sweep,
        "ops": ops(scored),
    }


# ---------------------------------------------------------------- sim change
def sim_actions(scored: list[Scored], th: dict[str, float]) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {"baseline_rules": [], "jev_only": [], "jev_plus_rules": []}
    for s in scored:
        ctx = s.row["context"]
        arms["baseline_rules"].append(sim_change.existing_rules(ctx)[0])
        hard = sim_change.hard_rules(ctx)
        if s.resp is None:
            arms["jev_only"].append("error")
            arms["jev_plus_rules"].append(hard[0] if hard else sim_change.fallback(ctx)[0])
            continue
        arms["jev_only"].append("hold" if s.resp.noul("suspicious") >= 0.5 else "allow")
        arms["jev_plus_rules"].append(hard[0] if hard else sim_change.policy(s.resp, ctx, th)[0])
    return arms


def fraud_metrics(gold: list[bool], acts: list[str]) -> dict[str, float]:
    caught = [a in ("hold", "step_up") for a in acts]
    held = [a == "hold" for a in acts]
    n_f, n_g = sum(gold), len(gold) - sum(gold)
    return {
        "fraud_caught_recall": sum(c and g for c, g in zip(caught, gold)) / max(1, n_f),
        "hold_precision": (sum(h and g for h, g in zip(held, gold)) / sum(held)) if any(held) else float("nan"),
        "genuine_friction_rate": sum(c and not g for c, g in zip(caught, gold)) / max(1, n_g),
        "genuine_held_rate": sum(h and not g for h, g in zip(held, gold)) / max(1, n_g),
    }


def choose_step_up(dev: list[Scored], base_th: dict[str, float]) -> tuple[float, list[dict]]:
    gold = [s.row["gold_fraud"] for s in dev]
    sweep = []
    for t in [x / 100 for x in range(5, 80, 5)]:
        m = fraud_metrics(gold, sim_actions(dev, {**base_th, "step_up": t})["jev_plus_rules"])
        sweep.append({"threshold": t, **m})
    ok = [r for r in sweep if r["fraud_caught_recall"] >= TARGET_FRAUD_RECALL]
    best = max(ok, key=lambda r: r["threshold"]) if ok else min(sweep, key=lambda r: r["threshold"])
    return best["threshold"], sweep


def eval_sim_change(client) -> dict[str, Any]:
    entry = catalogue.load()[sim_change.NAME]
    scored = ask_all(client, sim_change, entry, data.load("sim_change"))
    dev = [s for s in scored if s.row["split"] == "dev"]
    test = [s for s in scored if s.row["split"] == "test"]
    t, sweep = choose_step_up(dev, entry.thresholds)
    th = {**entry.thresholds, "step_up": t}
    gold = [s.row["gold_fraud"] for s in test]
    arms = {k: fraud_metrics(gold, v) for k, v in sim_actions(test, th).items()}
    ok = [s for s in test if s.resp]
    p = [s.resp.noul("suspicious") for s in ok]
    y = [s.row["gold_fraud"] for s in ok]
    # Calibration of a Noul: probability of "true" vs observed rate of fraud.
    bins = metrics.reliability(p, y)
    return {
        "use_case": sim_change.NAME,
        "n_dev": len(dev),
        "n_test": len(test),
        "fraud_rate_test": sum(gold) / len(gold),
        "chosen": {
            "step_up_threshold": t,
            "rule": f"highest threshold with dev fraud recall >= {TARGET_FRAUD_RECALL}",
            "catalogue_placeholder": entry.thresholds["step_up"],
        },
        "arms": arms,
        "auc_suspicious": metrics.auc(p, y),
        "calibration": {"ece": sum(b["n"] / len(p) * abs(b["acc"] - b["conf"]) for b in bins), "bins": bins},
        "sweep_dev": sweep,
        "ops": ops(scored),
    }


# ---------------------------------------------------------------- report
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def reliability_plot(bins: list[dict], title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], "o-", color="#ee0000")
    for b in bins:
        ax.annotate(str(b["n"]), (b["conf"], b["acc"]), fontsize=8, xytext=(3, -10), textcoords="offset points")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xlabel, ylabel="observed rate", title=title)
    return _png(fig)


def sweep_plot(sweep: list[dict], x: str, ys: list[str], chosen: float, title: str) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for y, c in zip(ys, ["#ee0000", "#111", "#888"]):
        ax.plot([r[x] for r in sweep], [r[y] for r in sweep], "o-", ms=3, label=y, color=c)
    ax.axvline(chosen, color="#ee0000", ls=":", lw=1)
    ax.set(xlabel=f"{x} (dev split)", ylim=(0, 1.02), title=title)
    ax.legend(fontsize=8)
    return _png(fig)


def _table(arms: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(arms.values())))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        f"<tr><td>{html.escape(a)}</td>" + "".join(f"<td>{v:.3f}</td>" for v in m.values()) + "</tr>"
        for a, m in arms.items()
    )
    return f"<table><tr><th>arm</th>{head}</tr>{body}</table>"


def render(summary: dict[str, Any]) -> str:
    ci, sc = summary["care_intent"], summary["sim_change"]
    mock = summary["backend"] == "mock"
    warn = (
        (
            "<p class='warn'>Backend is the offline <b>mock</b>. These numbers test the harness and policies, "
            "not Jev. Set TYPESAFE_API_KEY and rerun for real results.</p>"
        )
        if mock
        else ""
    )
    imgs = {
        "ci_rel": reliability_plot(ci["calibration"]["bins"], "Intent confidence (test)", "Jev choice confidence"),
        "ci_sweep": sweep_plot(
            ci["sweep_dev"],
            "threshold",
            ["route_precision", "automation_rate", "misroute_rate"],
            ci["chosen"]["route_threshold"],
            "Route threshold choice",
        ),
        "sc_rel": reliability_plot(sc["calibration"]["bins"], "Suspicious noul (test)", "Jev noul probability"),
        "sc_sweep": sweep_plot(
            sc["sweep_dev"],
            "threshold",
            ["fraud_caught_recall", "genuine_friction_rate"],
            sc["chosen"]["step_up_threshold"],
            "Step-up threshold choice",
        ),
    }

    def img(k):
        return f"<img src='data:image/png;base64,{imgs[k]}'>"

    def opsline(o):
        return (
            f"P50 {o['p50_ms']:.0f} ms, P95 {o['p95_ms']:.0f} ms, cost per 1k decisions "
            f"${o['cost_per_1k_usd']:.4f}, errors {o['errors']}"
        )

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Jev POC evaluation</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:24px auto;color:#111}}
h1{{border-bottom:3px solid #ee0000;padding-bottom:6px}} table{{border-collapse:collapse;margin:8px 0}}
td,th{{border-bottom:1px solid #ddd;padding:4px 10px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
.warn{{background:#fff4cc;padding:8px 12px;border-left:4px solid #c90}} img{{margin-right:12px}} .muted{{color:#666}}</style>
</head><body><h1>Jev POC evaluation</h1>
<p class='muted'>backend {summary["backend"]} · model {html.escape(summary["model"])} · frozen synthetic sets in data/eval ·
thresholds chosen on dev, reported on test</p>{warn}
<h2>Care intent routing ({ci["n_test"]} test items)</h2>
<p>Route threshold {ci["chosen"]["route_threshold"]:.2f} ({html.escape(ci["chosen"]["rule"])}; catalogue placeholder
{ci["chosen"]["catalogue_placeholder"]}). Intent ECE {ci["calibration"]["ece"]:.3f}. {opsline(ci["ops"])}.</p>
{_table(ci["arms"])}{img("ci_rel")}{img("ci_sweep")}
<h2>SIM-change risk gate ({sc["n_test"]} test items, fraud rate {sc["fraud_rate_test"]:.1%})</h2>
<p>Step-up threshold {sc["chosen"]["step_up_threshold"]:.2f} ({html.escape(sc["chosen"]["rule"])}; catalogue placeholder
{sc["chosen"]["catalogue_placeholder"]}). AUC {sc["auc_suspicious"]:.3f}, ECE {sc["calibration"]["ece"]:.3f}.
{opsline(sc["ops"])}.</p>
{_table(sc["arms"])}{img("sc_rel")}{img("sc_sweep")}
<h2>Reading the arms</h2><ul>
<li><b>baseline_rules</b>: today's keyword router or static fraud rules.</li>
<li><b>jev_only</b>: Jev's top answer used directly, no thresholds or hard rules.</li>
<li><b>jev_plus_rules</b>: the gateway policy. Hard rules first, Jev judgments next, confidence thresholds last.</li>
</ul></body></html>"""


def main(argv: list[str] | None = None) -> dict[str, Any]:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    client = get_client()
    summary = {
        "backend": client.backend,
        "model": client.model,
        "care_intent": eval_care_intent(client),
        "sim_change": eval_sim_change(client),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "report.html").write_text(render(summary))
    print(f"\n=== L8 Evaluation  [backend: {client.backend}] ===")
    for uc in ("care_intent", "sim_change"):
        print(f"{uc}: chosen {summary[uc]['chosen']}")
        for arm, m in summary[uc]["arms"].items():
            print(f"  {arm:15s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print(f"report: {OUT / 'report.html'}")
    return summary


if __name__ == "__main__":
    main()
