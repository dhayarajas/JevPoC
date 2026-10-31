import pytest

from jevpoc.client import JevError
from jevpoc.gateway.core import CircuitBreaker
from jevpoc.usecases import care_intent, sim_change


def test_decision_logged_with_state_hash_not_state(gateway):
    d = gateway.decide(care_intent.NAME, care_intent.EXAMPLES[0])
    row = gateway.log.recent(1)[0]
    assert row["decision_id"] == d.decision_id and len(row["state_hash"]) == 64
    assert "917-555" not in str(row) and "917-555" not in str(d.state)


def test_hard_rule_skips_jev(gateway):
    d = gateway.decide(sim_change.NAME, sim_change.EXAMPLES[3])
    assert d.action == "hold" and d.backend == "rules" and d.reason_code.startswith("hard_rule")


def test_timeout_falls_back_toward_review(gateway):
    ex = {**sim_change.EXAMPLES[0], "device_age_days": 500}  # existing rules say allow
    d = gateway.decide(sim_change.NAME, ex, timeout_ms=1)
    assert d.fallback_used and d.action == "step_up"


def test_breaker_opens_after_failures(gateway):
    for _ in range(5):
        gateway.decide(sim_change.NAME, sim_change.EXAMPLES[1], timeout_ms=1)
    d = gateway.decide(sim_change.NAME, sim_change.EXAMPLES[1])
    assert "circuit_open" in d.reason_code


def test_errors_never_become_negative_answers(gateway):
    class Broken:
        backend, model = "jev", "jev-1.14.0"

        def ask(self, *a, **k):
            raise JevError("HTTP 500")

    gateway.client = Broken()
    d = gateway.decide(care_intent.NAME, {"latest_message": "My phone screen is cracked"})
    assert d.fallback_used and d.answers == {}


def test_unpublished_entry_refused(gateway):
    import dataclasses

    e = gateway.entries[care_intent.NAME]
    gateway.entries[care_intent.NAME] = dataclasses.replace(e, status="draft")
    with pytest.raises(PermissionError):
        gateway.decide(care_intent.NAME, care_intent.EXAMPLES[0])


def test_breaker_half_opens_after_cooldown():
    b = CircuitBreaker(threshold=2, window_s=10, cooldown_s=5)
    b.record(False, now=0)
    b.record(False, now=1)
    assert not b.allow(now=2)
    assert b.allow(now=7)
    b.record(True, now=7)
    assert b.state == "closed"
