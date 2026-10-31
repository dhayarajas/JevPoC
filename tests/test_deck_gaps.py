"""Checks for the deck items closed after the first POC (slides 17, 27-33, 44, 46, 47, 49)."""

import json

import httpx
import pytest

from jevpoc import catalogue
from jevpoc.client import JevResponse
from jevpoc.masking import contains_pii, mask_text


def resp(answers):
    return JevResponse("m", answers, {}, 1.0, "mock")


def choice(c, conf):
    return {"type": "choice", "choice": c, "confidence": conf}


def noul(p):
    return {"type": "noul", "noul": p}


def score(s):
    return {"type": "score", "score": s}


def test_catalogue_thresholds_match_slides():
    e = catalogue.load()
    assert e["noc.triage"].thresholds["page_severity"] == 4.0
    assert e["noc.triage"].thresholds["page_confidence"] == 0.8
    assert e["care.answer_verify"].thresholds["promise_block"] == 0.1
    assert e["care.intent"].thresholds["clarify"] == 0.5
    assert len(e["care.intent"].questions["intent"]["criteria"]) == 40


def test_published_entries_have_approvals_and_versions():
    for e in catalogue.load().values():
        assert e.status == "published" and e.version == 2 and e.approvals and e.state_spec
        assert all(v.startswith(e.key) for v in e.question_versions.values())


def test_locked_catalogue_file_cannot_change(tmp_path):
    src = catalogue.CATALOGUE_DIR
    for p in src.glob("*"):
        (tmp_path / p.name).write_text(p.read_text())
    f = tmp_path / "care_intent_v2.yaml"
    f.write_text(f.read_text().replace("route: 0.75", "route: 0.70"))
    with pytest.raises(catalogue.CatalogueError):
        catalogue.load(tmp_path)


def test_unapproved_entry_cannot_publish(tmp_path):
    p = catalogue.CATALOGUE_DIR / "care_intent_v2.yaml"
    text = p.read_text().split("approvals:")[0] + "questions:" + p.read_text().split("\nquestions:")[1]
    (tmp_path / "x.yaml").write_text(text)
    with pytest.raises(catalogue.CatalogueError):
        catalogue.load(tmp_path)


def test_log_has_question_and_policy_versions(gateway):
    from jevpoc.usecases import care_intent

    d = gateway.decide("care.intent", care_intent.EXAMPLES[0])
    row = gateway.log.recent(1)[0]
    assert row["policy_version"].startswith("care.intent:") and row["policy_version"] == d.policy_version
    assert set(json.loads(row["question_versions"])) == {"intent", "in_scope", "needs_human", "language"}


def test_request_id_header_sent():
    from jevpoc.client import HttpJev

    seen = {}

    def handler(req):
        seen["rid"] = req.headers.get("x-request-id")
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {}, "usage": {}})

    HttpJev("k", transport=httpx.MockTransport(handler)).ask({}, {}, request_id="abc")
    assert seen["rid"] == "abc"


def test_spanish_routes_to_spanish_queue_and_other_language_to_agent():
    from jevpoc.usecases import care_intent

    th = catalogue.load()["care.intent"].thresholds
    base = {"intent": choice("billing_question", 0.9), "in_scope": noul(0.9), "needs_human": noul(0.1)}
    assert care_intent.policy(resp({**base, "language": choice("es", 0.9)}), {}, th)[0] == "route:billing_es"
    assert care_intent.policy(resp({**base, "language": choice("other", 0.9)}), {}, th)[0] == "agent"


def test_supervisor_alert_is_rate_limited():
    from jevpoc.usecases import escalation

    th = catalogue.load()["care.escalation"].thresholds
    r = resp(
        {
            "frustration": score(2.5),
            "churn_language": noul(0.7),
            "asks_for_human": noul(0.1),
            "bot_resolvable": noul(0.2),
        }
    )
    assert escalation.policy(r, {}, th)[0] == "alert:supervisor"
    action, _, details = escalation.policy(r, {"alerts_sent": 1}, th)
    assert action != "alert:supervisor" and details["alert_suppressed"]


def test_sim_jev_action_can_only_raise_caution():
    from jevpoc.usecases import sim_change

    th = catalogue.load()["fraud.sim_change"].thresholds
    low = {"suspicious": noul(0.05), "ato_risk": score(0.2), "pressure_tactics": noul(0.05)}
    assert sim_change.policy(resp({**low, "recommended_action": choice("allow", 0.99)}), {}, th)[0] == "allow"
    a, _, d = sim_change.policy(resp({**low, "recommended_action": choice("analyst", 0.9)}), {}, th)
    assert a == "hold" and d["notify"] == "fraud_analyst"


def test_noc_site_ids_are_hashed():
    from jevpoc.usecases import noc_triage

    ex = noc_triage.EXAMPLES[0]
    state = json.dumps(noc_triage.build_state(ex, catalogue.load()["noc.triage"]))
    for a in ex["alarms"]:
        assert a["element"].split("/")[0] not in state


def test_offer_cap_and_unsure_make_no_offer():
    from jevpoc.usecases import next_best_offer as nbo

    ex = dict(nbo.EXAMPLES[0])
    assert nbo.hard_rules({**ex, "offers_presented_30d": 2})[0] == "no_offer"
    th = catalogue.load()["sales.next_best_offer"].thresholds
    r = resp({"best_offer": choice("travelpass", 0.1), "open_to_offer": noul(0.9), "save_risk": score(0.1)})
    assert nbo.policy(r, ex, th)[0] == "no_offer"


def test_fairness_fields_never_reach_state():
    from jevpoc.usecases import next_best_offer as nbo

    ex = {**nbo.EXAMPLES[0], "usage": {"age": 61, "gender": "f", "data_gb": 40}}
    state = nbo.build_state(ex, catalogue.load()["sales.next_best_offer"])
    assert '"age"' not in json.dumps(state) and '"gender"' not in json.dumps(state)


def test_secrets_masked_and_destructive_actions_need_owner():
    from jevpoc.usecases import agent_guardrail as g

    t = mask_text("use token=abc123secretvalue and Bearer eyJhbGciOiJIUzI1NiJ9.x")
    assert "abc123secretvalue" not in t and "eyJhbGci" not in t and not contains_pii(t)
    ctx = {**g.EXAMPLES[0], "proposed_action": {"tool": "update_record", "args": {"op": "delete rows"}}}
    assert g.hard_rules(ctx) == ("require_approval:owner", "destructive_action")


def test_paired_bootstrap_and_coverage():
    from jevpoc.eval import metrics

    b = metrics.paired_bootstrap([True] * 90 + [False] * 10, [True] * 70 + [False] * 30)
    assert b["diff"] == pytest.approx(0.2) and b["ci_low"] > 0 and b["p_value"] < 0.05
    c = metrics.coverage_curve([0.9, 0.6, 0.2], [True, True, False])
    assert c[0]["coverage"] == 1.0 and c[-1]["coverage"] == 0.0


def test_frozen_sets_are_big_enough_with_temporal_spanish_and_relabel():
    from jevpoc.eval import data

    for name in data.SETS:
        rows = data.load(name)
        assert len(rows) >= 2000 and {r["period"] for r in rows} == {"train", "post_launch"}
        assert 0.04 <= sum("relabel" in r for r in rows) / len(rows) <= 0.06
        assert rows == data.SETS[name]()
    assert sum(r["lang"] == "es" for r in data.load("care_intent")) >= 150


def test_protocol_gate_and_robustness_run(tmp_path, monkeypatch):
    from jevpoc.eval import gate, harness, protocol, robustness

    for m in (harness, protocol, robustness):
        monkeypatch.setattr(m, "OUT", tmp_path)
    s = protocol.main([])
    assert set(s["pilots"]) == {"care_intent", "escalation", "answer_verify", "sim_change"}
    assert set(s["pilots"]["sim_change"]["arms"]) == {"A_today", "B_constrained_llm", "C_jev"}
    assert s["scorecard"]["rows"]["care_intent"]["privacy_contract"] == "red"
    assert gate.main([], report=tmp_path / "protocol.json") == 0
    worse = gate.extract(s)
    worse["care_intent"]["accuracy"] -= 0.05
    assert gate.compare(worse, gate.extract(s))
    monkeypatch.setattr(robustness, "N", 40)
    r = robustness.main([])
    assert r["injection"]["fraud.sim_change"]["less_cautious_after_injection"] <= 0.05
    assert all(p["max_state_tokens"] <= r["state_size"]["budget_tokens"] for p in r["state_size"]["sweep"])


def test_openai_compatible_llm_arm_parses_and_rejects():
    from jevpoc.eval.llm_arm import OpenAICompatLLM

    q = {"x": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}}}

    def ok(req):
        content = json.dumps({"x": {"answer": "b"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10}}
        )

    r = OpenAICompatLLM("http://llm", "k", "m", transport=httpx.MockTransport(ok)).ask({}, q)
    assert r.choice("x") == "b" and r.cost_usd > 0

    def bad(req):
        content = json.dumps({"x": {"answer": "z"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    from jevpoc.client import JevError

    with pytest.raises(JevError):
        OpenAICompatLLM("http://llm", "k", "m", transport=httpx.MockTransport(bad)).ask({}, q)
