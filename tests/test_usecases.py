import pytest

from jevpoc.client import JevResponse
from jevpoc.usecases import REGISTRY, agent_guardrail, answer_verify, care_intent, sim_change


@pytest.mark.parametrize("name", list(REGISTRY))
def test_every_example_runs(gateway, name):
    for ex in REGISTRY[name].EXAMPLES:
        d = gateway.decide(name, ex)
        assert d.action and d.reason_code


def resp(answers):
    return JevResponse("jev-1.14.0", answers, {}, 10, "jev")


def noul(p):
    return {"type": "noul", "noul": p}


def test_care_policy_thresholds():
    th = {"route": 0.75, "clarify": 0.45, "needs_human": 0.6, "in_scope_min": 0.3}
    base = {"in_scope": noul(0.9), "needs_human": noul(0.1)}

    def intent(conf):
        return {
            "type": "choice",
            "choice": "billing_question",
            "confidence": conf,
            "probabilities": {"billing_question": 0.6, "plan_change": 0.3},
        }

    assert care_intent.policy(resp({**base, "intent": intent(0.9)}), {}, th)[0] == "route:billing"
    assert care_intent.policy(resp({**base, "intent": intent(0.5)}), {}, th)[0].startswith("clarify:")
    assert care_intent.policy(resp({**base, "intent": intent(0.2)}), {}, th)[0] == "agent"
    assert care_intent.policy(resp({**base, "needs_human": noul(0.9), "intent": intent(0.9)}), {}, th)[0] == "agent"


def test_price_promise_blocked_by_code_even_if_jev_misses():
    ex = answer_verify.EXAMPLES[2]
    r = resp(
        {
            "consistent_with_facts": noul(0.99),
            "price_promise": noul(0.01),
            "answers_question": noul(0.9),
            "failure_type": {"type": "choice", "choice": "invented_credit", "confidence": 0.9},
        }
    )
    th = {"consistent_min": 0.9, "answers_min": 0.6, "promise_block": 0.3}
    assert answer_verify.policy(r, ex, th)[0] == "block:price_promise"


def test_wrong_amount_caught_in_code():
    assert answer_verify.unsupported_amounts("It is $65 a month", answer_verify.FACTS) == ["$65"]
    assert answer_verify.unsupported_amounts("It is $80 per line", answer_verify.FACTS) == []


def test_sim_features_do_the_maths():
    f = sim_change.features(sim_change.EXAMPLES[2])
    assert f["changes_72h_band"] == "2_plus" and f["geo_match"] is False and f["new_device"] is True
    assert "msisdn" not in sim_change.build_state(sim_change.EXAMPLES[2], None)


def test_guardrail_denylist_and_prod_write():
    assert agent_guardrail.hard_rules(agent_guardrail.EXAMPLES[3])[0] == "block"
    r = resp(
        {
            "safe_action": noul(0.99),
            "on_goal": noul(0.9),
            "looping": noul(0.0),
            "approval_level": {"type": "choice", "choice": "none", "confidence": 0.9},
        }
    )
    th = {"safe_min": 0.9, "on_goal_min": 0.5, "loop_stop": 0.6}
    assert agent_guardrail.policy(r, agent_guardrail.EXAMPLES[1], th)[0].startswith("require_approval")
    assert agent_guardrail.policy(r, agent_guardrail.EXAMPLES[0], th)[0] == "allow"


def test_next_best_offer_fallback_makes_no_offer():
    from jevpoc.usecases import next_best_offer

    for ex in next_best_offer.EXAMPLES:
        action, _ = next_best_offer.fallback(ex)
        assert action == "no_offer"
