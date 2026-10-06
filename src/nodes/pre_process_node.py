"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.caller_contract import is_screen_rejection, validate_caller_context
from src.services.runtime_settings import settings_from_state
from src.services.screening import contains_credentials, scan_for_threats


# The Marketplace runner seeds input_context with its own conversation history on every
# invocation (shared/bootstrap/marketplace_app.py); the caller neither sends that key nor can
# suppress it, and the build_input_context hook can only overwrite its value, never remove it.
# It is platform plumbing rather than caller data, so it is dropped here, before the caller
# contract runs: the unknown-field guard below stays strict for everything a caller can
# actually send, and no value screen is ever asked to judge a transcript that contains this
# agent's own earlier answers. The value may also be None, which this tolerates.
_PLATFORM_CONTEXT_KEYS = frozenset({"conversation_history"})


def _without_platform_context(raw: Any) -> Any:
    """The caller-supplied half of input_context, platform-injected keys removed."""
    if not isinstance(raw, dict):
        return raw
    return {k: v for k, v in raw.items() if k not in _PLATFORM_CONTEXT_KEYS}


# The platform input filter replaces personal-data shapes with this marker
# before any template code runs. In this domain it also fires on ordinary
# capitalised clinical terminology, so its presence is reported to the caller
# rather than passed off as a complete question.
_REDACTION_MARKER = "[MASKED]"


class PreProcessNode(FunctionNode):
    """Validate the caller's request before the clinical pipeline runs.

    Owns the caller-data contract for the whole agent:

      * trust — external callers must be verified; declared here rather than
        inherited, and matched by the deployment manifest;
      * the question — non-empty, within the configured length bound, and clear
        of instruction-injection patterns, protected clinical identifiers and
        credential shapes;
      * the structured context channel — validated against the declared
        contract, since the platform's own input filter does not scan it;
      * redaction transparency — when the platform filter has rewritten the
        question, that fact is recorded so the answer can say so.

    Rejections name the field and a stable reason code. The rejected text is
    never copied into the error log, the audit event, or the response.
    """

    # External callers must be verified — this is the agent's trust boundary.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        settings = settings_from_state(state)
        user_input = state.get("user_input", "")

        if not isinstance(user_input, str) or not user_input.strip():
            return self._reject(state, "empty_input", "user_input: must be a non-empty string")

        stripped = user_input.strip()

        max_chars = settings["query_max_chars"]
        if len(stripped) > max_chars:
            return self._reject(
                state,
                "input_too_long",
                f"user_input: exceeds the maximum of {max_chars} characters",
            )

        threat = scan_for_threats(stripped)
        if threat is not None:
            category, code = threat
            # silent: the message names the screen category that fired.
            return self._reject(state, code, f"user_input: refused by the input screen ({category})", silent=True)

        validated_context, context_error = validate_caller_context(
            _without_platform_context(state.get("input_context")),
            max_terms=settings["max_context_terms"],
        )
        if context_error is not None:
            # The context screen and the context CONTRACT share one error string. Only the
            # screen half names a detector, so only that half is withheld.
            return self._reject(
                state,
                "context_contract_violation",
                context_error,
                silent=is_screen_rejection(context_error),
            )

        # Defence in depth for the structured channel: the values are already
        # constrained to an inert alphabet, but a credential-shaped value
        # anywhere in a node's returned mapping fails the request at the
        # platform output gate with an error the caller cannot act on.
        if contains_credentials(validated_context):
            # silent: naming the credential screen tells a prober which check caught them.
            return self._reject(
                state,
                "context_credential_shape",
                "input_context: credential-shaped value refused",
                silent=True,
            )

        input_redacted = _REDACTION_MARKER in stripped
        channel = validated_context.get("channel", "unknown")

        emit_trace_event(
            "clinical_input_validated",
            {
                "input_length": len(stripped),
                "channel": channel,
                "context_terms": len(validated_context.get("clinical_terms", [])),
                "input_redacted": input_redacted,
                "source": "ClinicalBiomedicalQAAgent",
            },
            state,
        )
        return {
            "validated_input": stripped,
            "validated_context": validated_context,
            "input_redacted": input_redacted,
            "enriched_context": {
                "source": "ClinicalBiomedicalQAAgent",
                "channel": channel,
            },
            "status": AgentStatus.SUCCESS.value,
        }

    def _reject(self, state: dict[str, Any], reason: str, message: str, *, silent: bool = False) -> dict[str, Any]:
        """Refuse the request, naming the field and a stable reason code only.

        `silent` withholds the message from the caller. A SCREENED refusal names the
        detector that caught the payload, and returning that lets an attacker probe the
        screen one try at a time. Every other refusal names a field and a rule, which is
        what the caller needs to fix the request -- without it they get a blank spinner
        and cannot tell a rejection from a hang.
        """
        emit_trace_event(
            "input_validation_failed",
            {"reason": reason, "source": "PreProcessNode"},
            state,
        )
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
            # The runner surfaces `formatted_output or result` as `output`; error_log never
            # leaves the graph, so a reason left only there reaches no one.
            **(
                {}
                if silent
                else {"formatted_output": "Request could not be completed. " + f"PreProcessNode: {message}"}
            ),
        }
