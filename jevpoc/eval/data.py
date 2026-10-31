"""Synthetic, frozen eval sets.  python -m jevpoc.eval.data  rewrites data/eval/*.jsonl.

Labels come from the generator, never from a model. Each set is split once into
dev (for threshold choice) and test (reported once, never tuned on).
"""

from __future__ import annotations

import json
import random

from jevpoc.catalogue import ROOT

OUT = ROOT / "data" / "eval"

INTENT_TEMPLATES = {
    "billing_question": [
        "Why is my bill {amt} higher this month?",
        "There's a charge on my statement I don't understand.",
        "I was charged a fee twice, can you explain?",
        "what is this {amt} line item on my invoice",
        "My monthly amount went up and nobody told me why.",
    ],
    "payment_arrangement": [
        "Can I get an extension on my payment until the {day}?",
        "I can't pay the full amount this month, can I split it?",
        "Is it possible to move my due date to the {day}?",
        "I need a bit more time to pay, I get paid next Friday.",
    ],
    "plan_change": [
        "I want to switch plan to something cheaper.",
        "Can I downgrade my line to a basic plan?",
        "How do I add Disney+ perk to my plan?",
        "Move me to unlimited ultimate please.",
        "I'd like fewer features and a lower monthly price.",
    ],
    "device_tech_support": [
        "My phone won't turn on after the update.",
        "The screen is frozen and restart doesn't help.",
        "Battery drains in two hours since yesterday.",
        "Apps crash every time I open the camera.",
        "wifi calling stopped working on my iPhone",
    ],
    "network_coverage": [
        "I have no signal at my house in {city}.",
        "Calls keep dropping on my commute.",
        "Data is really slow downtown today, is there an outage?",
        "Only one bar inside my office building.",
        "Is something wrong with the towers near {city}? Nothing loads.",
    ],
    "activation_sim": [
        "How do I activate my new SIM card?",
        "I need to set up an eSIM on my new watch.",
        "I want to transfer my number to the phone I just bought.",
        "My new line isn't working, I think it needs activation.",
    ],
    "order_status": [
        "Where is my order? It hasn't arrived.",
        "Can you give me the tracking for the phone I bought?",
        "My delivery was supposed to come yesterday.",
        "Has my accessory shipped yet?",
    ],
    "international_roaming": [
        "I'm traveling to {country} next week, will my phone work?",
        "How much is TravelPass in {country}?",
        "Do I get data when I'm abroad?",
        "Going on a trip overseas, what do I need to turn on?",
    ],
    "cancel_service": [
        "I want to cancel my service.",
        "Please close my account, I'm moving to another carrier.",
        "I'm done with Verizon, how do I port out?",
        "I'm switching to T-Mobile, what do I owe to leave?",
    ],
    "fraud_unrecognised": [
        "There's a line on my account I didn't add.",
        "I don't recognize an order placed on my account.",
        "Someone else changed my SIM, I think I got hacked.",
        "I got a text about an upgrade that was not me.",
    ],
    "device_upgrade": [
        "Am I eligible for an upgrade yet?",
        "I want a new phone, what trade in deals are there?",
        "How much would I get to trade-in my old iPhone?",
        "Can I get the new Pixel now or do I need to wait?",
    ],
}
HUMAN = [
    "Let me talk to a real person.",
    "Agent please.",
    "Connect me with a representative, I'm not doing this with a bot.",
]
OUT_OF_SCOPE = [
    "What's the weather like tomorrow?",
    "Write me a poem about cats.",
    "Who won the football game last night?",
    "hey",
    "ok thanks",
]
FILL = {
    "amt": ["$20", "$35", "$50", "$12.99"],
    "day": ["15th", "20th", "1st"],
    "city": ["Tampa", "Queens", "Dallas"],
    "country": ["Mexico", "Italy", "Japan", "Canada"],
}

QUEUES_PREFIX = "route:"


def _fill(t: str, rng: random.Random) -> str:
    for k, v in FILL.items():
        t = t.replace("{" + k + "}", rng.choice(v))
    if rng.random() < 0.15:
        t = t.lower()
    if rng.random() < 0.1:
        t = t.replace("?", "??")
    return t


def care_intent(n: int = 480, seed: int = 7) -> list[dict]:
    from jevpoc.usecases.care_intent import QUEUES

    rng = random.Random(seed)
    rows = []
    intents = list(INTENT_TEMPLATES)
    for i in range(n):
        r = rng.random()
        if r < 0.06:
            msg, intent, gold = rng.choice(HUMAN), None, "agent"
        elif r < 0.10:
            msg, intent, gold = rng.choice(OUT_OF_SCOPE), None, "out_of_scope"
        else:
            intent = intents[i % len(intents)]
            msg = _fill(rng.choice(INTENT_TEMPLATES[intent]), rng)
            gold = QUEUES_PREFIX + QUEUES[intent]
        rows.append(
            {
                "id": f"ci-{i:04d}",
                "split": "dev" if i % 2 == 0 else "test",
                "context": {
                    "latest_message": msg,
                    "account": {
                        "plan": "Unlimited Plus",
                        "lines": rng.randint(1, 5),
                        "tenure_months": rng.randint(1, 150),
                    },
                },
                "gold_intent": intent,
                "gold_action": gold,
            }
        )
    return rows


def sim_change(n: int = 600, seed: int = 11, fraud_rate: float = 0.12) -> list[dict]:
    """Latent label first, then features drawn from overlapping label-conditional distributions."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        fraud = rng.random() < fraud_rate

        def p(a: float, b: float, fraud: bool = fraud) -> bool:
            return rng.random() < (a if fraud else b)

        ctx = {
            "event_type": rng.choices(
                ["sim_swap", "esim_transfer", "port_out"], [0.4, 0.3, 0.3] if fraud else [0.5, 0.4, 0.1]
            )[0],
            "channel": rng.choices(
                ["retail_store", "web", "app", "care_phone"],
                [0.1, 0.35, 0.2, 0.35] if fraud else [0.4, 0.25, 0.25, 0.1],
            )[0],
            "auth_method": rng.choices(
                ["id_scan", "sms_otp", "pin", "security_questions"],
                [0.05, 0.6, 0.15, 0.2] if fraud else [0.35, 0.4, 0.2, 0.05],
            )[0],
            "device_age_days": 0 if p(0.85, 0.5) else rng.randint(2, 900),
            "tenure_months": rng.randint(1, 150),
            "changes_72h": rng.choice([0, 1, 2, 3]) if p(0.6, 0.08) else 0,
            "geo_distance_km": rng.randint(200, 3000) if p(0.5, 0.06) else rng.randint(0, 40),
            "contacts_7d": rng.randint(4, 9) if p(0.45, 0.05) else rng.randint(0, 2),
            "interaction_notes": rng.choice(
                [
                    "Caller said it is urgent and they are traveling, lost my phone, hurry.",
                    "Asked to skip the PIN, said boss needs it right now.",
                    "Customer calm, answered all questions.",
                    "",
                ]
                if fraud
                else [
                    "Customer upgraded phone, calm.",
                    "Customer calm, answered all questions.",
                    "",
                    "",
                    "Phone broken, new phone from store.",
                ]
            ),
            "fraud_flag": fraud and rng.random() < 0.05,
        }
        rows.append(
            {"id": f"sc-{i:04d}", "split": "dev" if i % 2 == 0 else "test", "context": ctx, "gold_fraud": fraud}
        )
    return rows


def write() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("care_intent", care_intent()), ("sim_change", sim_change())):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {(OUT / name).relative_to(ROOT)}.jsonl")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(OUT / f"{name}.jsonl")]


if __name__ == "__main__":
    write()
