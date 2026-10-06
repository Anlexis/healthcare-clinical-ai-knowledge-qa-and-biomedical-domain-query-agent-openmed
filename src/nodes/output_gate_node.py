"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# Mandatory clinical disclaimer. Every answer this agent publishes must carry
# it: the response is reference information, and the final clinical judgement
# belongs to a physician.
CLINICAL_DISCLAIMER = "この回答は参考情報です。最終判断は医師にご確認ください。"

_DISCLAIMER_HEADING = "【免責事項】"


class OutputGateNode(FunctionNode):
    """Attach the mandatory clinical disclaimer to the assembled answer.

    This node APPLIES the disclaimer; it does not certify it. Verification
    happens once, at the publication boundary, against the value that is
    actually returned to the caller — a check placed here would only be able to
    re-read the string this method just built, which cannot fail and therefore
    proves nothing.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        clinical_response = str(state.get("clinical_response") or "")
        domain_routed = state.get("domain_routed", "")

        if clinical_response:
            formatted_response = f"{clinical_response}\n\n{_DISCLAIMER_HEADING}\n{CLINICAL_DISCLAIMER}"
        else:
            formatted_response = f"{_DISCLAIMER_HEADING}\n{CLINICAL_DISCLAIMER}"

        emit_trace_event(
            "clinical_disclaimer_applied",
            {
                "domain": domain_routed,
                "response_length": len(formatted_response),
            },
            state,
        )
        return {
            "formatted_response": formatted_response,
            "disclaimer_applied": True,
            "status": AgentStatus.SUCCESS.value,
        }
