"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.runtime_settings import settings_from_state

# Shown when the platform input filter rewrote part of the question. Without
# it the answer reads as a complete response to a question the reader asked,
# when in fact terms were removed before the pipeline saw them.
_REDACTION_NOTICE = (
    "■ ご注意: 入力フィルタにより質問の一部が伏せ字に置換されました。"
    "正確な回答のため、疾患名・薬剤名は構造化コンテキスト (clinical_terms) で指定してください。"
)


class ResponseFormatNode(FunctionNode):
    """Assemble the clinical answer from retrieved passages and guidance.

    Rendering limits come from the resolved runtime settings, so an operator
    changing a declared value changes the answer.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        settings = settings_from_state(state)
        validated_input = str(state.get("validated_input") or state.get("user_input") or "")
        retrieved_passages = state.get("retrieved_passages") or []
        guideline_matches = state.get("guideline_matches") or []
        code_candidates = state.get("icd11_codes") or []
        query_intent = state.get("query_intent", "general_biomedical")
        context = state.get("validated_context") or {}
        clinical_terms = list(context.get("clinical_terms") or [])
        input_redacted = bool(state.get("input_redacted", False))

        excerpt = validated_input[: settings["answer_query_excerpt_chars"]]
        response_parts = [f"【臨床情報回答】 クエリ: {excerpt}", ""]

        if input_redacted:
            response_parts.extend([_REDACTION_NOTICE, ""])

        if clinical_terms:
            # Rendered as one term line. Composing caller tokens is a hazard the
            # publication boundary owns: two terms that are individually inert
            # can form a protected-identifier shape once joined, which no
            # per-field input check can see. The boundary re-scans the finished
            # answer for exactly that reason.
            response_parts.append(f"■ 指定された臨床用語: {' '.join(clinical_terms)}")
            response_parts.append("")

        if retrieved_passages:
            response_parts.append("■ 参照情報（院内知識ベース）:")
            for index, passage in enumerate(retrieved_passages[: settings["retrieval_top_k"]], 1):
                response_parts.append(f"  {index}. {passage.get('text', '')}")
                response_parts.append(f"     出典: {passage.get('source', '')} ({passage.get('domain', '')})")
            response_parts.append("")

        if guideline_matches:
            response_parts.append("■ 参照ガイドライン:")
            for match in guideline_matches[: settings["max_guidelines"]]:
                response_parts.append(f"  - {match.get('title', '')} [{match.get('guideline_id', '')}]")
            response_parts.append("")

        if code_candidates:
            response_parts.append("■ 分類コード候補（未検証・質問文からの抽出）:")
            for candidate in code_candidates[: settings["max_icd11_codes"]]:
                response_parts.append(f"  - {candidate.get('code', '')}（未検証）")
            response_parts.append("")

        clinical_response = "\n".join(response_parts).strip()

        emit_trace_event(
            "response_formatted",
            {
                "intent": query_intent,
                "passage_count": len(retrieved_passages),
                "guideline_count": len(guideline_matches),
                "code_candidate_count": len(code_candidates),
                "input_redacted": input_redacted,
                "response_length": len(clinical_response),
            },
            state,
        )
        return {
            "clinical_response": clinical_response,
            "status": AgentStatus.SUCCESS.value,
        }
