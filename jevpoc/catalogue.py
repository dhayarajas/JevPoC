"""Question catalogue: versioned YAML files, one per use case (deck slide 44)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_DIR = ROOT / "catalogue"


@dataclass(frozen=True)
class Entry:
    use_case: str
    version: int
    owner: str
    status: str
    model: str
    timeout_ms: int
    state_budget_tokens: int
    questions: dict[str, dict[str, Any]]
    thresholds: dict[str, float]
    fallback: str
    path: str

    @property
    def key(self) -> str:
        return f"{self.use_case}@v{self.version}"

    def question_payload(self) -> dict[str, dict[str, Any]]:
        """Questions as sent to Jev: only type, instructions, criteria."""
        out = {}
        for qid, q in self.questions.items():
            out[qid] = {k: copy.deepcopy(v) for k, v in q.items() if k in ("type", "instructions", "criteria")}
        return out


def load(directory: Path = CATALOGUE_DIR) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    for p in sorted(directory.glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        e = Entry(
            use_case=d["use_case"],
            version=int(d["version"]),
            owner=d["owner"],
            status=d["status"],
            model=d["model"],
            timeout_ms=int(d.get("timeout_ms", 400)),
            state_budget_tokens=int(d.get("state_budget_tokens", 1500)),
            questions=d["questions"],
            thresholds=d.get("thresholds", {}),
            fallback=d.get("fallback", "rules"),
            path=str(p.relative_to(ROOT)),
        )
        prev = entries.get(e.use_case)
        if e.status == "published" and (prev is None or prev.status != "published" or e.version > prev.version):
            entries[e.use_case] = e
        elif prev is None:
            entries[e.use_case] = e
    return entries
