"""AgentCore Platform v1.0"""

# Carries the validated caller context across the subgraph boundary.
#
# The subgraph node contract passes exactly one value to the inner graph — the
# input string. Structured context is not forwarded, so an inner node reading
# ``state["input_context"]`` sees the empty mapping the inner graph built for
# itself, no matter what the caller sent.
#
# The outer node therefore publishes the already-validated context immediately
# before it invokes the inner graph, and the inner graph seeds it into its own
# initial state. Both happen in the same synchronous call, so a context
# variable scopes the hand-off correctly under concurrent invocations, where a
# module-level global would not.
#
# The publish is UNCONDITIONAL — an invocation with no structured context
# publishes the empty mapping — so a value can never survive from one
# invocation into the next.
#
# Only ALREADY-VALIDATED context is published here: the bridge is a transport,
# never a second entry point into the pipeline.

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_BOUNDARY_PAYLOAD: ContextVar[dict[str, Any]] = ContextVar("clinical_boundary_payload")

# Fields carried across the boundary, with the value used when none was
# published. Any outer field an inner node needs must be listed here: an inner
# node reading an outer key directly always sees the inner graph's own state,
# where that key does not exist.
BOUNDARY_FIELDS: dict[str, Any] = {
    "validated_context": {},
    "input_redacted": False,
}


def publish_boundary_payload(payload: dict[str, Any]) -> None:
    """Publish the validated outer fields for the inner graph about to run."""
    _BOUNDARY_PAYLOAD.set({key: payload.get(key, default) for key, default in BOUNDARY_FIELDS.items()})


def consume_boundary_payload() -> dict[str, Any]:
    """Return the published fields, or their defaults when none was published."""
    published = _BOUNDARY_PAYLOAD.get({})
    return {key: published.get(key, default) for key, default in BOUNDARY_FIELDS.items()}
