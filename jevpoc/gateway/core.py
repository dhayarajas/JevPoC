"""Decision gateway: catalogue -> state builder -> Jev -> policy -> log.

Every use case level (L2-L6), the HTTP service (L7) and the eval harness (L8)
go through Gateway.decide, so they share masking, timeout, breaker, fallback
and the decision log.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jevpoc import catalogue
from jevpoc.client import JevError, JevResponse, JevTimeout, get_client
from jevpoc.masking import contains_pii

DB_PATH = catalogue.ROOT / "reports" / "decision_log.sqlite"
RETENTION_DAYS = 395  # about 13 months (deck slide 47)


@dataclass
class Decision:
    decision_id: str
    use_case: str
    catalogue_version: int
    action: str
    reason_code: str
    model: str
    backend: str
    latency_ms: float
    fallback_used: bool
    state_hash: str
    state_tokens: int
    question_versions: dict[str, str] = field(default_factory=dict)
    policy_version: str = ""
    answers: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    state: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CircuitBreaker:
    """Opens after `threshold` failures within `window_s`; half-opens after `cooldown_s`."""

    def __init__(self, threshold: int = 5, window_s: float = 30, cooldown_s: float = 30):
        self.threshold, self.window_s, self.cooldown_s = threshold, window_s, cooldown_s
        self.failures: list[float] = []
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.opened_at is None:
                return True
            if now - self.opened_at >= self.cooldown_s:
                return True  # half-open probe
            return False

    def record(self, ok: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if ok:
                self.failures.clear()
                self.opened_at = None
                return
            self.failures = [t for t in self.failures if now - t < self.window_s] + [now]
            if len(self.failures) >= self.threshold:
                self.opened_at = now

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if time.monotonic() - self.opened_at >= self.cooldown_s else "open"


class DecisionLog:
    """One row per decision (deck slide 47). Stores the hash of masked state, never the state."""

    COLUMNS = (
        "decision_id",
        "ts",
        "use_case",
        "catalogue_version",
        "model",
        "backend",
        "state_hash",
        "state_tokens",
        "answers",
        "action",
        "reason_code",
        "latency_ms",
        "fallback_used",
        "caller",
        "human_override",
        "label",
        "question_versions",
        "policy_version",
    )

    def __init__(self, path: Path | str = DB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute(
            """create table if not exists decision_log (
                decision_id text primary key, ts real, use_case text, catalogue_version int,
                model text, backend text, state_hash text, state_tokens int, answers text,
                action text, reason_code text, latency_ms real, fallback_used int, caller text,
                human_override text, label text)"""
        )
        cols = {r[1] for r in self.db.execute("pragma table_info(decision_log)")}
        for c in ("question_versions", "policy_version"):
            if c not in cols:
                self.db.execute(f"alter table decision_log add column {c} text")
        self.db.execute("create index if not exists ix_uc_ts on decision_log(use_case, ts)")
        self.db.commit()
        self.purge()

    def purge(self, days: int = RETENTION_DAYS) -> int:
        with self._lock:
            n = self.db.execute("delete from decision_log where ts < ?", (time.time() - days * 86400,)).rowcount
            self.db.commit()
        return n

    def write(self, d: Decision, caller: str) -> None:
        row = (
            d.decision_id,
            time.time(),
            d.use_case,
            d.catalogue_version,
            d.model,
            d.backend,
            d.state_hash,
            d.state_tokens,
            json.dumps(d.answers),
            d.action,
            d.reason_code,
            d.latency_ms,
            int(d.fallback_used),
            caller,
            None,
            None,
            json.dumps(d.question_versions),
            d.policy_version,
        )
        with self._lock:
            self.db.execute(f"insert into decision_log values ({','.join('?' * len(row))})", row)
            self.db.commit()

    def recent(self, limit: int = 50, use_case: str | None = None) -> list[dict[str, Any]]:
        q = "select * from decision_log"
        args: tuple = ()
        if use_case:
            q += " where use_case = ?"
            args = (use_case,)
        q += " order by ts desc limit ?"
        with self._lock:
            rows = self.db.execute(q, args + (limit,)).fetchall()
        return [dict(zip(self.COLUMNS, r)) for r in rows]

    def stats(self) -> list[dict[str, Any]]:
        q = """select use_case, count(*), avg(latency_ms), sum(fallback_used), backend
               from decision_log group by use_case, backend order by use_case"""
        with self._lock:
            rows = self.db.execute(q).fetchall()
        return [
            {
                "use_case": r[0],
                "decisions": r[1],
                "avg_latency_ms": round(r[2] or 0, 1),
                "fallbacks": r[3],
                "backend": r[4],
            }
            for r in rows
        ]


def state_hash(state: Any) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def policy_version(module) -> str:
    src = inspect.getsource(module)
    return f"{module.NAME}:{hashlib.sha256(src.encode()).hexdigest()[:10]}"


def estimate_tokens(obj: Any) -> int:
    return max(1, len(json.dumps(obj, ensure_ascii=False)) // 4)


class Gateway:
    def __init__(self, client=None, log: DecisionLog | None = None, entries: dict[str, catalogue.Entry] | None = None):
        from jevpoc.usecases import REGISTRY

        self.client = client or get_client()
        self.log = log if log is not None else DecisionLog()
        self.entries = entries or catalogue.load()
        self.usecases = REGISTRY
        self.breakers: dict[str, CircuitBreaker] = {}

    def breaker(self, use_case: str) -> CircuitBreaker:
        return self.breakers.setdefault(use_case, CircuitBreaker())

    def decide(
        self, use_case: str, ctx: dict[str, Any], caller: str = "cli", timeout_ms: int | None = None
    ) -> Decision:
        if use_case not in self.entries:
            raise KeyError(f"unknown use case {use_case!r}")
        entry = self.entries[use_case]
        if entry.status != "published":
            raise PermissionError(f"{entry.key} is {entry.status}; only published entries are callable")
        uc = self.usecases[use_case]
        decision_id = str(uuid.uuid4())
        state = uc.build_state(ctx, entry)
        if contains_pii(state):
            raise ValueError("state builder leaked an identifier; refusing to send")
        tokens = estimate_tokens(state)
        base = dict(
            decision_id=decision_id,
            use_case=use_case,
            catalogue_version=entry.version,
            state_hash=state_hash(state),
            state_tokens=tokens,
            state=state,
            question_versions=entry.question_versions,
            policy_version=policy_version(uc),
        )

        hard = uc.hard_rules(ctx) if hasattr(uc, "hard_rules") else None
        if hard:
            d = Decision(
                action=hard[0],
                reason_code=f"hard_rule:{hard[1]}",
                model="none",
                backend="rules",
                latency_ms=0.0,
                fallback_used=False,
                **base,
            )
            self.log.write(d, caller)
            return d

        br = self.breaker(use_case)
        resp: JevResponse | None = None
        reason = None
        t0 = time.perf_counter()
        if not br.allow():
            reason = "fallback:circuit_open"
        else:
            try:
                resp = self.client.ask(
                    state,
                    uc.questions(entry, ctx),
                    entry.model,
                    timeout_s=(timeout_ms or entry.timeout_ms) / 1000,
                    request_id=decision_id,
                )
                br.record(True)
            except JevTimeout:
                br.record(False)
                reason = "fallback:timeout"
            except JevError as e:
                br.record(False)
                reason = f"fallback:error:{str(e)[:40]}"

        if resp is None:
            action, why = uc.fallback(ctx)
            d = Decision(
                action=action,
                reason_code=f"{reason}|{why}",
                model="none",
                backend="rules",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                fallback_used=True,
                **base,
            )
        else:
            action, why, details = uc.policy(resp, ctx, entry.thresholds)
            d = Decision(
                action=action,
                reason_code=why,
                model=resp.model,
                backend=resp.backend,
                latency_ms=resp.latency_ms,
                fallback_used=False,
                answers=resp.answers,
                details={**details, "input_tokens": resp.input_tokens, "cost_usd": resp.cost_usd},
                **base,
            )
        self.log.write(d, caller)
        return d
