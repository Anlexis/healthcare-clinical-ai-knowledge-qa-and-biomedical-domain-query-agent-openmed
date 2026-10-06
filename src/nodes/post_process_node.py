"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.nodes.output_gate_node import CLINICAL_DISCLAIMER
from src.services.screening import contains_credentials, scan_for_protected_identifiers
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

# Every state field that can carry answer text at this boundary. Anything added
# to this list must also be cleared on a refusal; a test pins the list against
# the node's own clearing behaviour so a new field cannot quietly join it.
OUTPUT_BEARING_FIELDS = ("result", "formatted_output")

# Returned in place of the answer when publication is refused. It must be
# non-empty: the framework resolves the caller-visible payload as the formatted
# output OR the raw result, so an empty notice would fall through to the very
# text being withheld.
WITHHELD_NOTICE = (
    "【回答を保留しました】この応答は公開基準を満たさなかったため差し止められました。内容は返却されません。"
)


class PostProcessNode(FunctionNode):
    """Publication boundary for the clinical answer.

    This is the single place where the published invariants are enforced, and
    it checks the value that is actually returned to the caller rather than one
    it assembled itself:

      * the mandatory clinical disclaimer is present;
      * no protected identifier appears in the answer — re-checked HERE because
        an identifier can enter after input validation, either from retrieved
        knowledge-base content or from the composition of individually-clean
        caller terms;
      * no credential shape appears in the answer, using the platform's own
        detector so this template can never be narrower than the gate that runs
        immediately after it.

    On a violation the answer is CLEARED and a non-empty notice is published in
    its place. Returning an error status alone would not withhold anything: the
    caller-visible payload falls back to the raw result field regardless of
    status, so a refusal that leaves that field populated ships the ungated
    answer inside the error envelope.

    Violation messages carry a reason code and a field name only. The matched
    text is never included — a credential-shaped string anywhere in this node's
    returned mapping makes the platform output gate raise, and the node wrapper
    then discards the whole mapping, clearing included.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        result = state.get("result") or ""
        answer = result if isinstance(result, str) else ""

        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=answer,
            domain="HCR ClinicalBiomedicalQAAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(answer, str) and not self._find_violation(answer + _review):
            answer = answer + _review

        violation = self._find_violation(answer)
        if violation is not None:
            return self._withhold(state, violation)

        emit_trace_event(
            "clinical_output_published",
            {
                "result_length": len(answer),
                "disclaimer_applied": bool(state.get("disclaimer_applied", False)),
            },
            state,
        )
        return {
            "result": answer,
            "formatted_output": answer,
            "status": AgentStatus.SUCCESS.value,
        }

    def _find_violation(self, answer: str) -> str | None:
        """Return a reason code for the first failed publication invariant."""
        if not answer:
            return "empty_answer"
        if CLINICAL_DISCLAIMER not in answer:
            return "disclaimer_missing"
        code = scan_for_protected_identifiers(answer)
        if code is not None:
            return code
        if contains_credentials(answer):
            return "credential.detected"
        return None

    def _withhold(self, state: dict[str, Any], reason: str) -> dict[str, Any]:
        """Refuse publication: clear every answer-bearing field, publish a notice.

        Every field in ``OUTPUT_BEARING_FIELDS`` is written explicitly. Omitting
        a key would not clear it — a partial update leaves the previous value in
        place, so an omitted key keeps whatever the pipeline had already
        written there.
        """
        emit_trace_event(
            "clinical_output_withheld",
            {"reason": reason, "field": "result", "source": "PostProcessNode"},
            state,
        )
        return {
            "result": "",
            "formatted_output": WITHHELD_NOTICE,
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PostProcessNode: publication refused (result: {reason})"],
        }
