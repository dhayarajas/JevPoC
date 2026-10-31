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
N = 2000  # deck slide 46: at least 2k labelled cases per use case
DAYS = 90
TRAIN_DAYS = 60  # temporal split: train period vs post-launch

ES_TEMPLATES = {
    "billing_question": ["¿Por qué mi factura subió {amt} este mes?", "Tengo un cargo en mi factura que no entiendo"],
    "network_coverage": ["No tengo señal en {city}, las llamadas se cortan", "El internet del teléfono está muy lento"],
    "device_tech_support": ["Mi teléfono no funciona, la pantalla está negra", "Mi teléfono se apaga solo"],
    "payment_arrangement": ["No puedo pagar todo este mes, ¿puedo pagar después?", "Quiero cambiar la fecha de pago"],
    "cancel_service": ["Quiero cancelar mi servicio por favor", "Voy a cerrar mi cuenta"],
    "international_roaming": ["Voy a viajar a {country}, ¿cómo uso mi teléfono allá?"],
}


def _meta(i: int, n: int, rng: random.Random, gold) -> dict:
    """Split, temporal period, and a 5% blind second label from a simulated second labeller."""
    day = i * DAYS // n
    period = "train" if day < TRAIN_DAYS else "post_launch"
    # dev (threshold choice) only sees the train period; every post-launch case is held out in test
    m = {"split": "dev" if period == "train" and i % 2 == 0 else "test", "day": day, "period": period}
    if i % 20 == 7:
        m["relabel"] = gold if rng.random() > 0.03 else "disagree"
    return m


def _fill(t: str, rng: random.Random) -> str:
    for k, v in FILL.items():
        t = t.replace("{" + k + "}", rng.choice(v))
    if rng.random() < 0.15:
        t = t.lower()
    if rng.random() < 0.1:
        t = t.replace("?", "??")
    return t


def care_intent(n: int = N, seed: int = 7) -> list[dict]:
    from jevpoc.usecases.care_intent import QUEUES

    rng = random.Random(seed)
    rows = []
    intents = list(INTENT_TEMPLATES)
    for i in range(n):
        r = rng.random()
        lang = "en"
        if r < 0.10:
            lang = "es"
            intent = rng.choice(list(ES_TEMPLATES))
            msg = _fill(rng.choice(ES_TEMPLATES[intent]), rng)
            gold = QUEUES_PREFIX + QUEUES[intent] + "_es"
        elif r < 0.16:
            msg, intent, gold = rng.choice(HUMAN), None, "agent"
        elif r < 0.20:
            msg, intent, gold = rng.choice(OUT_OF_SCOPE), None, "out_of_scope"
        else:
            intent = intents[i % len(intents)]
            msg = _fill(rng.choice(INTENT_TEMPLATES[intent]), rng)
            gold = QUEUES_PREFIX + QUEUES[intent]
        rows.append(
            {
                "id": f"ci-{i:04d}",
                **_meta(i, n, rng, gold),
                "lang": lang,
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


def sim_change(n: int = N, seed: int = 11, fraud_rate: float = 0.12) -> list[dict]:
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
        rows.append({"id": f"sc-{i:04d}", **_meta(i, n, rng, fraud), "context": ctx, "gold_fraud": fraud})
    return rows


ESC_CALM = [
    "How do I check my data usage?",
    "Thanks, that helped.",
    "Can you tell me my due date?",
    "Ok, I will try that now.",
    "This is a bit annoying but fine, what next?",
]
ESC_POOLS = {
    "frustrated": [
        "This is the third time this month, it's ridiculous, nobody fixes it.",
        "Unacceptable. I have been waiting for an hour and this is useless.",
        "Still broken. I am so fed up with this.",
    ],
    "churn": [
        "I'm going to cancel and switch to T-Mobile.",
        "I'm done with Verizon, close my account.",
        "AT&T is cheaper, I'm leaving.",
    ],
    "human": ["Let me talk to a real person.", "Get me an agent now.", "I want a supervisor."],
}


def escalation(n: int = N, seed: int = 13) -> list[dict]:
    """Gold says whether the conversation should leave the bot (handoff or alert)."""
    rng = random.Random(seed)
    classes = ["calm", "calm", "frustrated", "churn", "human"]
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        turns = []
        for _ in range(rng.randint(1, 3)):
            turns.append({"role": "customer", "text": rng.choice(ESC_CALM)})
            turns.append(
                {
                    "role": "bot",
                    "text": rng.choice(["Let me check that.", "Try restarting the device.", "Here is the link."]),
                }
            )
        last = rng.choice(ESC_CALM if cls == "calm" else ESC_POOLS[cls])
        turns.append({"role": "customer", "text": last})
        gold = cls != "calm"
        ctx = {"turns": turns, "tenure_months": rng.randint(1, 150), "hold_count": rng.randint(0, 3)}
        rows.append(
            {"id": f"es-{i:04d}", **_meta(i, n, rng, gold), "context": ctx, "class": cls, "gold_escalate": gold}
        )
    return rows


def answer_verify(n: int = N, seed: int = 17) -> list[dict]:
    """Gold says whether the LLM draft is safe to send unchanged."""
    from jevpoc.usecases.answer_verify import EXAMPLES, FACTS

    rng = random.Random(seed)
    q_plan = EXAMPLES[0]["customer_question"]
    pools = {
        "good": [
            (q_plan, EXAMPLES[0]["draft_reply"]),
            (q_plan, f"Your plan is {FACTS['plan']} at {FACTS['monthly_price']}."),
            ("What plan am I on?", f"You are on {FACTS['plan']}."),
        ],
        "wrong_facts": [
            (q_plan, EXAMPLES[1]["draft_reply"]),
            (q_plan, "Unlimited Plus is $70 per line and includes 25 GB of premium hotspot."),
            ("What plan am I on?", "You are on Unlimited Welcome."),
        ],
        "price_promise": [
            (EXAMPLES[2]["customer_question"], EXAMPLES[2]["draft_reply"]),
            ("Why is there a late fee?", "No worries, I will waive the $10 late fee for you."),
            ("Can I get a discount?", "I can give you $20 off next month as a courtesy."),
        ],
    }
    classes = ["good", "good", "wrong_facts", "price_promise"]
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        q, draft = rng.choice(pools[cls])
        gold = cls == "good"
        ctx = {"customer_question": q, "draft_reply": draft, "facts": FACTS}
        rows.append({"id": f"av-{i:04d}", **_meta(i, n, rng, gold), "context": ctx, "class": cls, "gold_send": gold})
    return rows


SETS = {"care_intent": care_intent, "sim_change": sim_change, "escalation": escalation, "answer_verify": answer_verify}


def write() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in SETS.items():
        rows = fn()
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {(OUT / name).relative_to(ROOT)}.jsonl")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(OUT / f"{name}.jsonl")]


if __name__ == "__main__":
    write()
