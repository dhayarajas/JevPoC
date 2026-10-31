"""Question catalogue: versioned YAML files, one per use case (deck slide 44)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_DIR = ROOT / "catalogue"
LOCK_FILE = "LOCK.json"


class CatalogueError(RuntimeError):
    pass


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
    state_spec: str = ""
    approvals: tuple = ()
    eval: dict | None = None

    @property
    def key(self) -> str:
        return f"{self.use_case}@v{self.version}"

    @property
    def question_versions(self) -> dict[str, str]:
        """Content hash of each catalogue question."""
        return self.versions_of(self.question_payload())

    def versions_of(self, questions: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Content hash of each question as sent, logged with every decision."""
        return {qid: f"{self.key}:{qid}:{_digest(json.dumps(q, sort_keys=True))}" for qid, q in questions.items()}

    def question_payload(self) -> dict[str, dict[str, Any]]:
        """Questions as sent to Jev: only type, instructions, criteria."""
        out = {}
        for qid, q in self.questions.items():
            out[qid] = {k: copy.deepcopy(v) for k, v in q.items() if k in ("type", "instructions", "criteria")}
        return out


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def _read_lock(directory: Path) -> dict[str, str]:
    f = directory / LOCK_FILE
    return json.loads(f.read_text()) if f.exists() else {}


def lock(directory: Path = CATALOGUE_DIR) -> dict[str, str]:
    """Record the hash of every published or retired file. Locked files may never change."""
    locked = _read_lock(directory)
    for p in sorted(directory.glob("*.yaml")):
        d = yaml.safe_load(p.read_text())
        if d["status"] in ("published", "retired"):
            locked.setdefault(p.name, _digest(p.read_text()))
    (directory / LOCK_FILE).write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n")
    return locked


def load(directory: Path = CATALOGUE_DIR) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    locked = _read_lock(directory)
    for p in sorted(directory.glob("*.yaml")):
        text = p.read_text()
        if p.name in locked and locked[p.name] != _digest(text):
            raise CatalogueError(f"{p.name} changed after it was locked; publish a new version instead")
        d = yaml.safe_load(text)
        if d["status"] == "published" and not d.get("approvals"):
            raise CatalogueError(f"{p.name} is published without approvals")
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
            path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            state_spec=d.get("state_spec", ""),
            approvals=tuple(d.get("approvals") or ()),
            eval=d.get("eval"),
        )
        prev = entries.get(e.use_case)
        if e.status == "published" and (prev is None or prev.status != "published" or e.version > prev.version):
            entries[e.use_case] = e
        elif prev is None:
            entries[e.use_case] = e
    return entries


if __name__ == "__main__":
    if sys.argv[1:] == ["lock"]:
        print(json.dumps(lock(), indent=2))
    else:
        for e in load().values():
            print(f"{e.key:34s} {e.status:10s} {e.model}  {e.path}")
