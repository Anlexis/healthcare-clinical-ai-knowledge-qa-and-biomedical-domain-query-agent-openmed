"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# When the agent runs behind the platform gateway, that gateway calls
# agent.invoke() directly and this module is not involved.

import os
import secrets
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from framework.utils.config_loader import load_config
from pydantic import BaseModel, Field
from shared.secrets import factory as secrets_factory

from src.graph.graph import ClinicalBiomedicalQAAgent
from src.services.caller_contract import drop_unknown_context_fields

app = FastAPI(title="Agent")

# Runtime parameters live in config/config.yaml and are handed to the graph as
# Graph(config=...) — the same convention the platform registry uses. Without
# this the standalone process would run with an empty configuration, and every
# declared runtime value would be silently ignored.
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config: dict[str, Any] = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}

_secrets_provider = secrets_factory(namespace="hcr", agent_name="ClinicalBiomedicalQAAgent")

agent = ClinicalBiomedicalQAAgent(config=_config)
agent.compile()
agent.provision_secrets(_secrets_provider)


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured context channel. Only the declared fields survive; see
    # src/services/caller_contract.py for the contract they must satisfy.
    input_context: dict[str, Any] = Field(default_factory=dict)


def _bearer_matches(supplied: str, expected: str) -> bool:
    """Constant-time bearer comparison that is safe for non-ASCII header input."""
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel,
    authorization: str,
    invoke_auth_token: str | None,
    internal_runner_token: str | None,
) -> TrustLevel:
    """Authenticate standalone callers without allowing external-token elevation.

    The internal runner token is a distinct deployment credential: it is
    considered only for an otherwise-anonymous caller and maps exactly to the
    internal level. The invoke token maps to the verified-external level, which
    is what this agent's manifest requires. Trust already established by
    middleware in front of this process is never changed.
    """
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


def _screen_context_for_credentials(context: dict[str, Any]) -> None:
    """Refuse a request whose structured context carries a credential shape.

    The first backbone node returns the structured context verbatim in its
    result, and the platform output gate scans every value of every result — so
    a credential-shaped string here fails the request inside the first node,
    with a traceback and no indication of which field was at fault. The request
    cannot succeed either way, so it is refused HERE with a message the caller
    can act on.

    The platform's own detector is used, so the set refused here is exactly the
    set the gate blocks. It scans values, never keys, and is defined as the
    union over a mapping's values — which is why scanning field by field is
    equivalent to scanning the whole mapping, and is what allows the offending
    field to be named. The value itself is never echoed.
    """
    for position, (name, value) in enumerate(context.items(), start=1):
        if not detect_credentials_in_value(value):
            continue
        safe = name.isascii() and name.replace("_", "").replace("-", "").isalnum()
        label = f"input_context.{name}" if safe else f"input_context field #{position}"
        # 400, not 422: the validation framework owns 422 and answers it with a
        # list of error objects, so reusing it makes client handling ambiguous.
        raise HTTPException(status_code=400, detail=f"{label}: credential-shaped value refused")


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    # This adapter is the entry-point auth boundary. Both tokens are
    # deployment-level caller credentials rather than agent secrets: no
    # invocation context exists before this boundary, so the per-request secret
    # binding cannot apply to them.
    trust = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )

    # Undeclared keys are DROPPED, not ignored. A validator that merely ignores
    # an unknown key leaves it in state, where the first backbone node returns
    # it verbatim into its result — dropping is what makes the channel safe by
    # construction rather than by convention.
    context = drop_unknown_context_fields(req.input_context)
    _screen_context_for_credentials(context)

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        response: dict[str, Any] = agent.invoke(req.input, ctx=ctx, input_context=context)
        return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "ClinicalBiomedicalQAAgent"}
