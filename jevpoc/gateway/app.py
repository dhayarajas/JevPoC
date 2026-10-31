"""L7 Decision gateway as an HTTP service (deck slide 43). Channels call this, never the vendor.

uvicorn jevpoc.gateway.app:app --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from jevpoc.gateway.core import Gateway

STATIC = Path(__file__).parent / "static"


class DecideRequest(BaseModel):
    context: dict[str, Any]
    caller: str = "api"
    timeout_ms: int | None = Field(default=None, ge=1, le=10_000)


class Chaos(BaseModel):
    fail_rate: float = Field(0.0, ge=0, le=1)
    slow_rate: float = Field(0.0, ge=0, le=1)


def create_app(gateway: Gateway | None = None) -> FastAPI:
    gw = gateway or Gateway()
    app = FastAPI(title="Jev decision gateway (POC)", version="0.1.0")
    app.state.gateway = gw

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "backend": gw.client.backend,
            "model": gw.client.model,
            "breakers": {uc: gw.breaker(uc).state for uc in gw.entries},
            "chaos": {
                "fail_rate": getattr(gw.client, "fail_rate", None),
                "slow_rate": getattr(gw.client, "slow_rate", None),
            },
        }

    @app.get("/v1/catalogue")
    def catalogue() -> list[dict[str, Any]]:
        return [
            {
                "use_case": e.use_case,
                "version": e.version,
                "owner": e.owner,
                "status": e.status,
                "model": e.model,
                "timeout_ms": e.timeout_ms,
                "fallback": e.fallback,
                "thresholds": e.thresholds,
                "questions": e.question_payload(),
                "path": e.path,
            }
            for e in gw.entries.values()
        ]

    @app.get("/v1/examples")
    def examples() -> dict[str, list[dict[str, Any]]]:
        return {name: getattr(m, "EXAMPLES", []) for name, m in gw.usecases.items()}

    @app.post("/v1/decide/{use_case}")
    def decide(use_case: str, req: DecideRequest) -> dict[str, Any]:
        try:
            return gw.decide(use_case, req.context, caller=req.caller, timeout_ms=req.timeout_ms).to_dict()
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except PermissionError as e:
            raise HTTPException(403, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/v1/decisions")
    def decisions(limit: int = 50, use_case: str | None = None) -> list[dict[str, Any]]:
        return gw.log.recent(min(limit, 500), use_case)

    @app.get("/v1/stats")
    def stats() -> list[dict[str, Any]]:
        return gw.log.stats()

    @app.post("/v1/chaos")
    def chaos(c: Chaos) -> dict[str, Any]:
        if gw.client.backend != "mock":
            raise HTTPException(400, "fault injection is only available on the mock backend")
        gw.client.fail_rate, gw.client.slow_rate = c.fail_rate, c.slow_rate
        return health()

    return app


app = create_app()
