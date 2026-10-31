"""CI regression gate (deck slide 46): no Arm C metric may get worse by more than 1 point.

python -m jevpoc.eval.gate            # compare reports/eval/protocol.json with the frozen baseline
python -m jevpoc.eval.gate --update   # accept the current run as the new baseline (review in the PR)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jevpoc.catalogue import ROOT
from jevpoc.eval.harness import OUT

BASELINE = ROOT / "data" / "eval" / "gate_baseline.json"
TOLERANCE = 0.01
HIGHER_IS_BETTER = ("accuracy", "macro_f1", "automated")
LOWER_IS_BETTER = ("ece", "false_action_rate")


def extract(summary: dict) -> dict[str, dict[str, float]]:
    out = {}
    for name, r in summary["pilots"].items():
        c = r["arms"]["C_jev"]
        out[name] = {k: round(c[k], 4) for k in HIGHER_IS_BETTER + LOWER_IS_BETTER if k in c}
    return out


def compare(current: dict, baseline: dict, tol: float = TOLERANCE) -> list[str]:
    fails = []
    for name, base in baseline.items():
        cur = current.get(name, {})
        for k, b in base.items():
            if k not in cur:
                fails.append(f"{name}.{k}: missing")
            elif k in HIGHER_IS_BETTER and cur[k] < b - tol:
                fails.append(f"{name}.{k}: {cur[k]:.4f} < baseline {b:.4f} - {tol}")
            elif k in LOWER_IS_BETTER and cur[k] > b + tol:
                fails.append(f"{name}.{k}: {cur[k]:.4f} > baseline {b:.4f} + {tol}")
    return fails


def main(argv: list[str] | None = None, report: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args(argv)
    summary = json.loads((report or OUT / "protocol.json").read_text())
    current = extract(summary)
    key = f"{summary['backend_c']}:{summary['model']}"
    baselines = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    if args.update:
        baselines[key] = current
        BASELINE.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
        print(f"baseline for {key} updated")
        return 0
    if key not in baselines:
        print(f"no baseline for {key}; run with --update and commit {BASELINE.name}")
        return 1
    fails = compare(current, baselines[key])
    for f in fails:
        print("REGRESSION", f)
    print("gate:", "FAIL" if fails else "pass", f"({key})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
