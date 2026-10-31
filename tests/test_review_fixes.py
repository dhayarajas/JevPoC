"""Regression tests for the Devin Review findings on PR #3."""

import json

from jevpoc import catalogue
from jevpoc.client import JevResponse
from jevpoc.gateway.core import DecisionLog, Gateway
from jevpoc.masking import contains_pii, mask


class Fixed:
    backend = "fixed"

    def __init__(self, answers):
        self.answers = answers

    def ask(self, state, questions, model=None, timeout_s=5.0, request_id=None):
        self.sent = questions
        return JevResponse(model, self.answers, {}, 1.0, self.backend)


def test_offer_question_version_follows_eligible_offers(tmp_path):
    from jevpoc.usecases import next_best_offer as nbo

    gw = Gateway(log=DecisionLog(tmp_path / "l.sqlite"))
    ex = nbo.EXAMPLES[0]
    a = gw.decide("sales.next_best_offer", ex).question_versions["best_offer"]
    b = gw.decide("sales.next_best_offer", {**ex, "owned": nbo.eligible(ex)[:1]}).question_versions["best_offer"]
    assert a != b


def test_uncertain_language_goes_to_agent():
    from jevpoc.usecases import care_intent

    th = catalogue.load()["care.intent"].thresholds
    r = JevResponse(
        "m",
        {
            "intent": {"type": "choice", "choice": "billing_question", "confidence": 0.9},
            "in_scope": {"type": "noul", "noul": 0.9},
            "needs_human": {"type": "noul", "noul": 0.1},
            "language": {"type": "choice", "choice": "es", "confidence": 0.5},
        },
        {},
        1.0,
        "mock",
    )
    assert care_intent.policy(r, {}, th)[:2] == ("agent", "language_uncertain")


def test_supervisor_alert_once_per_call_id(tmp_path):
    from jevpoc.usecases import escalation

    answers = {
        "frustration": {"type": "score", "score": 2.5},
        "churn_language": {"type": "noul", "noul": 0.8},
        "asks_for_human": {"type": "noul", "noul": 0.1},
        "bot_resolvable": {"type": "noul", "noul": 0.2},
    }
    gw = Gateway(client=Fixed(answers), log=DecisionLog(tmp_path / "l.sqlite"))
    ctx = {**escalation.EXAMPLES[0], "call_id": "c1"}
    first, second = gw.decide("care.escalation", ctx), gw.decide("care.escalation", ctx)
    assert first.action == "alert:supervisor" and first.details["alerts_sent"] == 1
    assert second.action != "alert:supervisor" and second.details["alert_suppressed"]
    assert gw.decide("care.escalation", {**ctx, "call_id": "c2"}).action == "alert:supervisor"


def test_clarify_counts_only_when_gold_intent_offered():
    from jevpoc.eval import protocol

    th = catalogue.load()["care.intent"].thresholds
    row = {"context": {"latest_message": "x"}, "gold_action": "route:fraud", "gold_intent": "fraud_unrecognised"}

    def r(top, second):
        return JevResponse(
            "m",
            {
                "intent": {
                    "type": "choice",
                    "choice": top,
                    "confidence": 0.6,
                    "probabilities": {top: 0.6, second: 0.3, "plan_change": 0.1},
                },
                "in_scope": {"type": "noul", "noul": 0.9},
                "needs_human": {"type": "noul", "noul": 0.1},
            },
            {},
            1.0,
            "mock",
        )

    assert not protocol._care(row, r("billing_question", "device_upgrade"), th, "C").correct
    assert protocol._care(row, r("billing_question", "fraud_unrecognised"), th, "C").correct


def test_dev_split_never_sees_post_launch():
    from jevpoc.eval import data

    for name in data.SETS:
        rows = data.load(name)
        assert all(r["period"] == "train" for r in rows if r["split"] == "dev")
        assert any(r["period"] == "train" for r in rows if r["split"] == "test")


def test_dict_form_secrets_are_masked_and_detected():
    raw = {"tool": "call_api", "args": {"api_key": "abc123", "Authorization": "Basic Zm9vOmJhcg==", "n": 3}}
    assert contains_pii(raw)
    m = mask(raw)
    assert "abc123" not in json.dumps(m) and "Zm9vOmJhcg" not in json.dumps(m) and not contains_pii(m)
    assert m["args"]["n"] == 3


def test_noc_site_ids_scrubbed_from_text_and_changes():
    from jevpoc.usecases import noc_triage

    ctx = {
        "alarms": [{"time": "1", "element": "BOS0101/router-a", "text": "BOS0101 link down, peer NYC0412 ok"}],
        "recent_changes": ["CR-1 on BOS0101 at 10:00", {"note": "rollback BOS0101"}],
    }
    s = json.dumps(noc_triage.build_state(ctx, catalogue.load()["noc.triage"]))
    assert "BOS0101" not in s and "NYC0412" not in s and "CR-1" in s
