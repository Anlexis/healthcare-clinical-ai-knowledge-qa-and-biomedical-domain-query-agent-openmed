"""Backbone invoke-order verification.

Asserts that a full invocation by a caller holding the trust level the manifest
declares runs the fixed five-node backbone in order and returns a real answer.

The caller context is built at the VERIFIED_EXTERNAL level — never an internal
one. An internal context satisfies every node in the graph and would hide a
node that demands more trust than an external caller can ever hold.
"""

import pytest
import yaml
from pathlib import Path

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

REPO_ROOT = Path(__file__).parents[2]

# The subgraph node occupying the `main` backbone slot.
MAIN_SLOT_NODE = "BiomedicalQAGraphNode"

# A clinical question that reaches a successful answer: non-empty, no protected
# identifier, a recognised intent.
VALID_PAYLOAD = "What are the drug-drug interactions for warfarin and aspirin per MHLW guidelines?"

EXPECTED_BACKBONE = [
    "InitializeNode",
    "PreProcessNode",
    MAIN_SLOT_NODE,
    "PostProcessNode",
    "FinalizeNode",
]


@pytest.fixture(autouse=True)
def _patch_all_audit(monkeypatch):
    """Mute audit emission at every node module — never stub shared.* wholesale."""
    for module in [
        "src.nodes.pre_process_node",
        "src.nodes.post_process_node",
        "src.nodes.query_parse_node",
        "src.nodes.domain_route_node",
        "src.nodes.biobert_retrieve_node",
        "src.nodes.clinical_kb_lookup_node",
        "src.nodes.response_format_node",
        "src.nodes.output_gate_node",
    ]:
        monkeypatch.setattr(f"{module}.emit_trace_event", lambda *a, **k: None)
    monkeypatch.setattr("framework.nodes.base_node.emit_trace_event", lambda *a, **k: None)


def _agent():
    from src.graph.graph import ClinicalBiomedicalQAAgent

    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    agent = ClinicalBiomedicalQAAgent(config=config)
    agent.compile()
    return agent


class TestBackboneInvokeOrder:
    def test_backbone_invoke_order_success(self):
        agent = _agent()
        result = agent.invoke(
            user_input=VALID_PAYLOAD,
            ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL),
        )

        assert result["status"] == AgentStatus.SUCCESS.value, result.get("error_log", [])
        assert result["output"]
        assert "この回答は参考情報です" in str(result["output"])
        assert result["node_history"] == EXPECTED_BACKBONE

    def test_external_caller_context_is_the_one_exercised(self):
        agent = _agent()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = agent.invoke(user_input=VALID_PAYLOAD, ctx=ctx)

        assert ctx.caller_trust_level == TrustLevel.VERIFIED_EXTERNAL
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_an_anonymous_caller_is_refused_and_publishes_nothing(self):
        agent = _agent()
        result = agent.invoke(
            user_input=VALID_PAYLOAD,
            ctx=InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS),
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert not result["output"]
        assert "PostProcessNode" not in result["node_history"]
