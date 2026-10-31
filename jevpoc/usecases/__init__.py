"""Use cases, keyed by catalogue `use_case` id."""

from jevpoc.usecases import (
    agent_guardrail,
    answer_verify,
    care_intent,
    escalation,
    next_best_offer,
    noc_triage,
    sim_change,
)

REGISTRY = {
    m.NAME: m
    for m in (care_intent, escalation, answer_verify, sim_change, noc_triage, next_best_offer, agent_guardrail)
}
