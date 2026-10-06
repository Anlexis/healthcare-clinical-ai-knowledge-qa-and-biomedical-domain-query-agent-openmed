"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# Query intent -> biomedical knowledge domain.
_INTENT_TO_DOMAIN = {
    "drug_interaction": "chemicals",
    "disease_classification": "diseases",
    "treatment_guideline": "diseases",
    "molecular_biology": "genes",
    "oncology": "oncology",
    "anatomy": "anatomy",
    "general_biomedical": "diseases",
}
_DEFAULT_DOMAIN = "diseases"


class DomainRouteNode(FunctionNode):
    """Route the question to the matching biomedical knowledge domain."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query_intent = state.get("query_intent", "general_biomedical")
        extracted_entities = state.get("extracted_entities") or []

        domain = _INTENT_TO_DOMAIN.get(query_intent, _DEFAULT_DOMAIN)

        emit_trace_event(
            "domain_routed",
            {
                "intent": query_intent,
                "domain": domain,
                "entity_count": len(extracted_entities),
            },
            state,
        )
        return {
            "domain_routed": domain,
            "status": AgentStatus.SUCCESS.value,
        }
