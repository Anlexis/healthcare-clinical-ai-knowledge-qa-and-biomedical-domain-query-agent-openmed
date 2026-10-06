"""AgentCore Platform v1.0"""

import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.screening import scan_for_threats

__all__ = ["QueryParseNode", "scan_for_threats"]

# Intent keywords, checked in order. Oncology is tested before the general
# treatment-guideline bucket so "treatment options for cancer" routes to the
# oncology knowledge domain rather than the guideline one.
_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("drug_interaction", ("drug interaction", "薬物相互作用", "相互作用", "drug-drug")),
    ("disease_classification", ("icd", "disease code", "疾患コード", "診断コード")),
    ("oncology", ("oncology", "cancer", "腫瘍", "がん", "tumor")),
    ("treatment_guideline", ("treatment", "guideline", "治療", "ガイドライン", "mhlw", "厚生労働省")),
    ("molecular_biology", ("gene", "protein", "遺伝子", "タンパク")),
    ("anatomy", ("anatomy", "解剖", "organ", "臓器")),
)

# Candidate classification codes: an uppercase letter followed by up to four
# alphanumerics, optionally with a sub-classification suffix.
_CODE_CANDIDATE = re.compile(r"\b[A-Z][A-Z0-9]{1,4}(?:\.[A-Z0-9]+)?\b")


class QueryParseNode(FunctionNode):
    """Classify the clinical question and extract candidate codes.

    Re-runs the input screen. The backbone already screened the same text, so
    this is defence in depth: it keeps the inner workflow safe when it is
    exercised on its own, and it re-checks the value that the pipeline actually
    carries rather than the one the request arrived with.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        validated_input = str(state.get("validated_input") or state.get("user_input") or "")

        threat = scan_for_threats(validated_input)
        if threat is not None:
            category, code = threat
            emit_trace_event(
                "query_parse_rejection",
                {"category": category, "reason": code, "action": "blocked", "source": "QueryParseNode"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"QueryParseNode: query refused by the input screen ({category})"],
            }

        query_lower = validated_input.lower()
        intent = "general_biomedical"
        for candidate, keywords in _INTENT_KEYWORDS:
            if any(keyword in query_lower for keyword in keywords):
                intent = candidate
                break

        entities = [
            {"type": "classification_code_candidate", "value": code}
            for code in _CODE_CANDIDATE.findall(validated_input)
        ]

        emit_trace_event(
            "query_parsed",
            {
                "intent": intent,
                "entity_count": len(entities),
                "input_length": len(validated_input),
            },
            state,
        )
        return {
            "query_intent": intent,
            "extracted_entities": entities,
            "status": AgentStatus.SUCCESS.value,
        }
