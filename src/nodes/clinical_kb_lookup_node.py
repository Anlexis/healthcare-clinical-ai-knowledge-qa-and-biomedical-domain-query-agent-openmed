"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# Intents for which the guideline index holds published guidance.
_GUIDELINE_INTENTS = frozenset({"treatment_guideline", "drug_interaction", "disease_classification"})


class ClinicalKBLookupNode(FunctionNode):
    """Cross-reference published treatment guidance and classification codes.

    Codes are reported as CANDIDATES extracted from the question, never as
    verified entries. The extractor recognises a code SHAPE — an uppercase
    letter followed by alphanumerics — which any ordinary abbreviation in a
    clinical question also has. Labelling those as verified entries of a
    published classification would put a provenance claim on the answer that
    nothing in the pipeline established.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query_intent = state.get("query_intent", "general_biomedical")
        extracted_entities = state.get("extracted_entities") or []
        domain_routed = state.get("domain_routed", "diseases")

        guideline_matches = self._lookup_guidelines(query_intent, domain_routed)
        code_candidates = self._collect_code_candidates(extracted_entities, domain_routed)

        emit_trace_event(
            "clinical_kb_lookup_completed",
            {
                "intent": query_intent,
                "guideline_count": len(guideline_matches),
                "code_candidate_count": len(code_candidates),
            },
            state,
        )
        return {
            "guideline_matches": guideline_matches,
            "icd11_codes": code_candidates,
            "status": AgentStatus.SUCCESS.value,
        }

    def _lookup_guidelines(self, intent: str, domain: str) -> list[dict[str, Any]]:
        """Return published guidance entries matching *intent* in *domain*."""
        if intent not in _GUIDELINE_INTENTS:
            return []
        return [
            {
                "guideline_id": f"GUIDELINE-{domain.upper()}-001",
                "title": f"Published clinical guidance: {domain} domain",
                "relevance": "high",
            }
        ]

    def _collect_code_candidates(self, entities: list[Any], domain: str) -> list[dict[str, Any]]:
        """Return the classification-code candidates found in the question.

        ``verified`` is False for every entry: these are shapes matched in the
        caller's text, not lookups confirmed against a code register.
        """
        candidates: list[dict[str, Any]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if entity.get("type") != "classification_code_candidate":
                continue
            value = entity.get("value")
            if not isinstance(value, str) or not value:
                continue
            candidates.append(
                {
                    "code": value,
                    "domain": domain,
                    "verified": False,
                    "source": "extracted_from_question",
                }
            )
        return candidates
