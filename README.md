# JevPoC

Proof of concept for TypeSafe AI's **Jev** (System One) on Verizon decision points. Jev answers typed questions (Choice, Score, Noul) about a state and returns probabilities. Customer-facing text stays with an LLM or a template. Money, entitlement and fraud rules stay in Verizon code.

The ladder runs from one raw API call to a gateway with an evaluation harness. Each level is one command.

| Level | What it shows | Command |
|---|---|---|
| L1 | Hello Jev. One ticket, one Choice, one Score, one Noul, raw request and response | `make l1` |
| L2 | Care intent routing. Choice over 11 intents, route / clarify / agent by confidence | `make l2` |
| L3 | Escalation gate. Frustration Score and churn Nouls re-scored every customer turn | `make l3` |
| L4 | Answer verification. LLM draft checked against KB facts, price promises blocked, one regeneration | `make l4` |
| L5 | SIM-change and port-out risk gate. Masked state, hard rules first, allow / step-up / hold | `make l5` |
| L6 | NOC alarm triage, next-best-offer, internal AI-agent guardrail | `make l6` |
| L7 | Decision gateway. Catalogue, masking, timeout, circuit breaker, fallback, decision log, HTTP API and dashboard | `make l7`, `make serve` |
| L8 | Evaluation. Today's rules vs Jev alone vs Jev plus rules, on frozen labelled sets | `make l8` |

## Setup

```bash
make setup          # python3 -m venv .venv && pip install -r requirements.txt
make all            # runs L1 to L8
make test lint
```

With no key set, every level runs against `jevpoc/mock.py`, an offline stand-in with the same request and response shapes. Output is tagged `[backend: mock]`. Mock numbers check the pipeline and policies only. They say nothing about Jev's accuracy.

To use real Jev, create a key at console.typesafe.ai and export it:

```bash
export TYPESAFE_API_KEY=...        # never commit it, .env is git-ignored
make l1                            # output now shows [backend: jev]
make l8                            # real results in reports/eval/report.html
```

Other settings are in `.env.example`. The model is pinned to `jev-1.14.0` in every catalogue entry. `jev-latest` is never used in decision paths.

## How a decision flows

```
channel (IVR, chat, retail, NOC, agent runtime)
   |  raw context
   v
Gateway.decide(use_case, context)                 jevpoc/gateway/core.py
   1. catalogue entry, published only             catalogue/*.yaml
   2. state builder: mask PII, band numbers,      jevpoc/usecases/*.py
      trim to the token budget
   3. hard rules (deny lists, locks, flags)       -> decision without Jev
   4. Jev call with per-use-case timeout          jevpoc/client.py
      circuit breaker open, timeout or error      -> rules fallback, fails toward review
   5. policy: thresholds on Jev probabilities     -> action + reason code
   6. decision log row with state hash            reports/decision_log.sqlite
```

Rules the code follows, from the TypeSafe guidance and the Verizon POV:

- Counting, dates, distances and amounts are computed in the state builder and passed as bands. Jev is not asked to do arithmetic.
- Broad questions are split into atomic judgments, then combined in code.
- A failed or slow call is treated as "no answer", never as "no". Fallbacks fail toward more human review, never toward more risk.
- MSISDN, account number, IMEI, email and names never leave the gateway. The log stores a hash of the masked state, not the state.
- Prices and credits come from the pricing table and policy code, never from Jev.

## The gateway service (L7)

```bash
make serve                         # http://localhost:8000
```

The dashboard at `/` lets you pick a use case and example, edit the raw context, see the action, the Jev answers behind it and the masked state that was sent, inject errors or slow responses (mock only) and watch the circuit breaker and decision log.

API:

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/decide/{use_case}` | `{"context": {...}, "caller": "ivr", "timeout_ms": 400}` returns the decision |
| GET | `/v1/catalogue` | published question sets, thresholds, owners |
| GET | `/v1/examples` | sample contexts per use case |
| GET | `/v1/decisions` | latest decision log rows |
| GET | `/v1/stats` | volume, latency and fallbacks per use case |
| GET | `/v1/health` | backend, model, breaker state |
| POST | `/v1/chaos` | mock only, set error and slow-response rates |

## Evaluation (L8)

`data/eval/` holds two frozen synthetic sets with labels from the generator (`python -m jevpoc.eval.data` rebuilds them). Each is split once into dev and test. Thresholds are chosen on dev with a stated rule (route precision at least 0.95, fraud recall at least 0.95) and then reported once on test.

The report at `reports/eval/report.html` has, per use case:

- three arms: `baseline_rules` (today's keyword router or static fraud rules), `jev_only`, `jev_plus_rules` (the gateway policy)
- routing: action accuracy, automation rate, misroute rate, route precision, intent macro-F1
- fraud: fraud caught, hold precision, friction on genuine customers, AUC
- calibration: ECE and a reliability curve, plus the threshold sweep on dev
- operations: P50 and P95 latency, cost per 1,000 decisions, error count

The synthetic sets prove the harness. Before any go or no-go call, the same harness should run on a frozen sample of masked Verizon traffic labelled by Verizon reviewers.

## Layout

```
catalogue/                 versioned question sets per use case (owner, status, model, timeout, thresholds)
jevpoc/client.py           HTTP client for /v1/systemone and the response helpers
jevpoc/mock.py             offline backend
jevpoc/masking.py          PII and CPNI masking
jevpoc/usecases/           state builder, hard rules, policy, fallback and examples per use case
jevpoc/gateway/core.py     decide(), circuit breaker, decision log
jevpoc/gateway/app.py      FastAPI service and dashboard
jevpoc/eval/               data generator, metrics, harness and HTML report
jevpoc/run.py              the L1 to L8 runner
tests/                     pytest suite (runs offline)
```

## Adding a use case

1. Write `catalogue/<name>_v1.yaml` with the questions and placeholder thresholds.
2. Add `jevpoc/usecases/<name>.py` with `NAME`, `build_state`, `questions`, `policy`, `fallback`, optional `hard_rules` and `EXAMPLES`.
3. Register it in `jevpoc/usecases/__init__.py`.
4. Add labelled cases to the eval set and pick thresholds on dev before setting `status: published`.
