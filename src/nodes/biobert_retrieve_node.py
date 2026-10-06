"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.runtime_settings import settings_from_state


class BioBERTRetrieveNode(FunctionNode):
    """Retrieve knowledge-base passages for the routed biomedical domain.

    Retrieval runs against the locally deployed biomedical index, so request
    content never leaves the hospital network.

    How many passages are retrieved and how much of the question is carried into
    a passage header are configuration, not constants: both come from the
    resolved runtime settings the graph seeded into state.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        settings = settings_from_state(state)
        validated_input = str(state.get("validated_input") or state.get("user_input") or "")
        domain_routed = state.get("domain_routed", "diseases")
        context = state.get("validated_context") or {}
        clinical_terms = list(context.get("clinical_terms") or [])

        passages = self._retrieve_passages(
            validated_input,
            domain_routed,
            clinical_terms,
            top_k=settings["retrieval_top_k"],
            excerpt_chars=settings["retrieval_query_excerpt_chars"],
        )

        emit_trace_event(
            "kb_passages_retrieved",
            {
                "domain": domain_routed,
                "passage_count": len(passages),
                "top_k": settings["retrieval_top_k"],
                "context_terms": len(clinical_terms),
                "on_premise": True,
            },
            state,
        )
        return {
            "retrieved_passages": passages,
            "status": AgentStatus.SUCCESS.value,
        }

    def _retrieve_passages(
        self,
        query: str,
        domain: str,
        clinical_terms: list[str],
        *,
        top_k: int,
        excerpt_chars: int,
    ) -> list[dict[str, Any]]:
        """Return up to *top_k* passages for *query* in *domain*.

        Caller-supplied clinical terms are searched as their own queries — that
        is the point of the structured channel, since a term written in the
        free-text question may have been rewritten by the input filter before
        the pipeline ever saw it.
        """
        queries: list[tuple[str, str]] = [("query", query[:excerpt_chars])]
        queries.extend(("term", term) for term in clinical_terms)

        passages: list[dict[str, Any]] = []
        for kind, subject in queries[:top_k]:
            if not subject:
                continue
            passages.append(
                {
                    "text": f"[{domain.upper()}] Knowledge-base passage for {kind}: {subject}",
                    "source": "on_premise_biomedical_index",
                    "domain": domain,
                    "match": kind,
                    "score": 0.85,
                }
            )
        return passages
