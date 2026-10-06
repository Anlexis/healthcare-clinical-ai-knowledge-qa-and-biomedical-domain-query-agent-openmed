"""Publication-boundary tests.

Two things are proved here.

First, that the published invariants are enforced on the value that is actually
returned to the caller: the mandatory clinical disclaimer, no protected
identifier, no credential shape.

Second, that a refusal WITHHOLDS. The framework resolves the caller-visible
payload as the formatted output OR the raw result, with no status check, so a
refusal that only sets an error status still ships the ungated answer. Each
clearing assertion therefore checks PRESENCE AND EMPTINESS — the key must be in
the returned update, and its value must be the cleared one. Asserting only
``not result.get(field)`` would pass on a gate that clears nothing, because a
partial update that omits a key leaves the previous value in place.
"""

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.output_gate_node import CLINICAL_DISCLAIMER
from src.nodes.post_process_node import OUTPUT_BEARING_FIELDS, WITHHELD_NOTICE, PostProcessNode

ANSWER_BODY = "【臨床情報回答】 クエリ: warfarin interactions"
CLEAN_ANSWER = f"{ANSWER_BODY}\n\n【免責事項】\n{CLINICAL_DISCLAIMER}"


@pytest.fixture(autouse=True)
def _patch_audit(monkeypatch):
    for module in ("src.nodes.post_process_node", "src.nodes.output_gate_node"):
        monkeypatch.setattr(f"{module}.emit_trace_event", lambda *a, **k: None)


def _publish(answer):
    return PostProcessNode().execute({"result": answer, "disclaimer_applied": True})


def _assert_withheld(result, released_text):
    """Every answer-bearing field is present in the update and cleared."""
    assert result["status"] == AgentStatus.ERROR.value
    for field in OUTPUT_BEARING_FIELDS:
        assert field in result, f"{field} omitted — an omitted key leaves the old value in state"
    assert result["result"] == ""
    assert result["formatted_output"] == WITHHELD_NOTICE
    assert released_text not in str(result)


class TestCleanPathControl:
    """A refuse-everything gate would pass every withholding test below, so the
    same request must still produce its real answer."""

    def test_a_compliant_answer_is_published_unchanged(self):
        result = _publish(CLEAN_ANSWER)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == CLEAN_ANSWER
        assert result["result"] == CLEAN_ANSWER

    def test_the_withheld_notice_is_truthy(self):
        """An empty notice would fall through to the raw result — producing the
        exact leak the notice exists to prevent."""
        assert bool(WITHHELD_NOTICE)
        assert WITHHELD_NOTICE.strip() != ""


class TestWithholdingPaths:
    def test_missing_disclaimer_is_withheld(self):
        _assert_withheld(_publish(ANSWER_BODY), ANSWER_BODY)

    def test_protected_identifier_in_the_answer_is_withheld(self):
        leaked = f"{ANSWER_BODY}\n患者番号：12345678\n\n【免責事項】\n{CLINICAL_DISCLAIMER}"
        _assert_withheld(_publish(leaked), "12345678")

    def test_composed_identifier_from_inert_terms_is_withheld(self):
        """Two caller terms that are individually inert can form an identifier
        once rendered together; per-field input validation cannot see that."""
        leaked = f"■ 指定された臨床用語: pt 98765\n\n【免責事項】\n{CLINICAL_DISCLAIMER}"
        _assert_withheld(_publish(leaked), "98765")

    def test_credential_shape_in_the_answer_is_withheld(self):
        leaked = f"{ANSWER_BODY} AKIA" + f"IOSFODNN7EXAMPLE\n\n【免責事項】\n{CLINICAL_DISCLAIMER}"
        result = _publish(leaked)
        assert result["status"] == AgentStatus.ERROR.value
        assert result["result"] == ""
        assert result["formatted_output"] == WITHHELD_NOTICE
        assert "AKIA" not in str(result)

    def test_empty_answer_is_withheld(self):
        result = _publish("")
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"] == WITHHELD_NOTICE


class TestRefusalMessages:
    def test_the_message_carries_a_reason_code_and_a_field_only(self):
        result = _publish(f"{ANSWER_BODY}\n患者番号：12345678\n{CLINICAL_DISCLAIMER}")
        message = result["error_log"][0]
        assert "result" in message
        assert "12345678" not in message
        assert "患者番号" not in message

    def test_no_source_path_or_traceback_reaches_the_surface(self):
        rendered = str(_publish(ANSWER_BODY))
        assert "Traceback" not in rendered
        assert ".py" not in rendered


class TestFieldInventory:
    def test_every_answer_bearing_field_is_declared_and_cleared(self):
        """Pins the inventory: a new answer-bearing field cannot quietly join
        the state without also being cleared here."""
        result = _publish(ANSWER_BODY)
        assert set(OUTPUT_BEARING_FIELDS) == {"result", "formatted_output"}
        assert set(OUTPUT_BEARING_FIELDS) <= set(result)

    def test_the_published_answer_fields_are_the_only_ones_the_node_writes(self):
        published = _publish(CLEAN_ANSWER)
        assert set(published) == {"result", "formatted_output", "status"}


class TestDisclaimerApplication:
    def test_the_inner_node_applies_the_disclaimer_the_boundary_requires(self):
        from src.nodes.output_gate_node import OutputGateNode

        result = OutputGateNode().execute({"clinical_response": "Body text.", "domain_routed": "diseases"})
        assert CLINICAL_DISCLAIMER in result["formatted_response"]
        assert _publish(result["formatted_response"])["status"] == AgentStatus.SUCCESS.value
