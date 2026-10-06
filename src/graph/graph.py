"""AgentCore Platform v1.0"""

from typing import Any, ClassVar

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.graph.context_bridge import publish_boundary_payload
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import State
from src.services.runtime_settings import (
    RUNTIME_SETTINGS_KEY,
    resolve_runtime_settings,
)

# Runtime keys forwarded from the agent configuration to the inner workflow.
# Everything the inner nodes tune lives under one of these blocks; forwarding by
# name keeps the inner graph's configuration surface explicit instead of handing
# it the whole agent config.
_FORWARDED_CONFIG_KEYS = ("query", "retrieval", "clinical")


class BiomedicalQAGraphNode(GraphNode):
    """Wraps the biomedical workflow graph; occupies the `main` backbone slot.

    Subgraph node contract:
      get_subgraph()  — build the inner workflow with the forwarded configuration
      extract_input() — hand the validated question to the inner graph, and
                        publish the validated caller context across the boundary
      merge_output()  — map the inner result back onto the outer state
    """

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = dict(config or {})

    def get_subgraph(self) -> Any:
        """Instantiate the inner workflow graph with the forwarded configuration."""
        from src.graph.domain_workflow_graph import BiomedicalDomainWorkflowGraph

        return BiomedicalDomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the validated question and publish the outer boundary fields.

        The publish happens here because this is the last point that runs before
        the inner graph is invoked, and it runs on EVERY invocation so nothing
        can survive from a previous one.
        """
        publish_boundary_payload(
            {
                "validated_context": state.get("validated_context") or {},
                "input_redacted": bool(state.get("input_redacted", False)),
            }
        )
        return str(state.get("validated_input") or "")

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map the inner result back onto the outer state.

        The answer text is surfaced only when the inner workflow reports
        success. On any other inner status the answer field is written empty
        rather than omitted: a partial update that omits a key leaves the
        previous value in place, so omission would not clear anything.
        """
        status = sub_result.get("status")
        succeeded = status == AgentStatus.SUCCESS.value or status == AgentStatus.SUCCESS
        return {
            "result": sub_result.get("output") if succeeded else "",
            "status": status,
            # Compliance provenance for the audit trail. Without this the outer
            # backbone can only read its own state, where the inner workflow's
            # fields do not exist — so the audit record would report "no
            # disclaimer applied" on every successful clinical answer.
            "disclaimer_applied": bool(sub_result.get("disclaimer_applied", False)),
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the runtime configuration blocks the inner workflow reads."""
        return {key: self._config[key] for key in _FORWARDED_CONFIG_KEYS if key in self._config}


class ClinicalBiomedicalQAAgent(AgentBaseGraph):
    """Clinical knowledge Q&A and biomedical domain query agent.

    The fixed five-node backbone (initialize -> pre_process -> main ->
    post_process -> finalize) is wired by the framework. Domain complexity is
    encapsulated in the main-slot subgraph node, which wraps the biomedical
    workflow graph.

    Designed for on-premise deployment: knowledge-base access stays inside the
    hospital network.
    """

    @property
    def name(self) -> str:
        return "ClinicalBiomedicalQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        """Validate the backbone configuration and this agent's runtime settings.

        Runs before graph construction, so a malformed declared value fails at
        start-up rather than degrading silently to a default at request time.
        """
        super()._validate_config()
        resolve_runtime_settings(self.config)

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the resolved runtime settings into every invocation's state."""
        return {RUNTIME_SETTINGS_KEY: resolve_runtime_settings(self.config)}

    def register_nodes(self) -> None:
        super().register_nodes()  # injects the framework initialize / finalize nodes
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = BiomedicalQAGraphNode(config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.
