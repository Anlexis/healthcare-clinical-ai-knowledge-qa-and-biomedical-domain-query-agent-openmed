"""AgentCore Platform v1.0"""

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import consume_boundary_payload
from src.nodes.query_parse_node import QueryParseNode
from src.nodes.domain_route_node import DomainRouteNode
from src.nodes.biobert_retrieve_node import BioBERTRetrieveNode
from src.nodes.clinical_kb_lookup_node import ClinicalKBLookupNode
from src.nodes.response_format_node import ResponseFormatNode
from src.nodes.output_gate_node import OutputGateNode
from src.schemas.state import State
from src.services.runtime_settings import (
    RUNTIME_SETTINGS_KEY,
    resolve_runtime_settings,
)


class BiomedicalDomainWorkflowGraph(BaseGraph):
    """Inner domain workflow for the clinical Q&A pipeline.

    Six-node linear topology:
      START -> query_parse -> domain_route -> biobert_retrieve
            -> clinical_kb_lookup -> response_format -> output_gate -> END

    Built by the main-slot subgraph node. Inner nodes run at the caller's own
    trust level; the trust gate for external callers sits on the outer backbone.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "biomedical_domain_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Reject malformed runtime settings before the graph is built."""
        resolve_runtime_settings(self.config)

    # ── Initial state ─────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the resolved settings and the caller context into inner state.

        The subgraph node contract forwards only the input string, so the
        validated caller context arrives over the boundary bridge instead.
        """
        return {
            RUNTIME_SETTINGS_KEY: resolve_runtime_settings(self.config),
            **consume_boundary_payload(),
        }

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register the six domain nodes."""
        self._nodes["query_parse"] = QueryParseNode()
        self._nodes["domain_route"] = DomainRouteNode()
        self._nodes["biobert_retrieve"] = BioBERTRetrieveNode()
        self._nodes["clinical_kb_lookup"] = ClinicalKBLookupNode()
        self._nodes["response_format"] = ResponseFormatNode()
        self._nodes["output_gate"] = OutputGateNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear six-step clinical Q&A pipeline."""
        self._sg.add_edge(START, "query_parse")
        self._sg.add_edge("query_parse", "domain_route")
        self._sg.add_edge("domain_route", "biobert_retrieve")
        self._sg.add_edge("biobert_retrieve", "clinical_kb_lookup")
        self._sg.add_edge("clinical_kb_lookup", "response_format")
        self._sg.add_edge("response_format", "output_gate")
        self._sg.add_edge("output_gate", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing hook required by the base contract.

        The topology is linear, so no conditional edge calls this. It is
        annotated with this graph's own state type because the graph engine
        reads a path callable's annotation as its input schema and projects
        away any field the annotation does not declare.
        """
        return END if state.get("status") == AgentStatus.ERROR.value else "output_gate"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the result handed back to the subgraph node.

        The answer text is surfaced only on success, so a refused or failed run
        cannot hand a partially-gated answer back across the boundary.
        """
        status = state.get("status")
        succeeded = status == AgentStatus.SUCCESS.value or status == AgentStatus.SUCCESS
        answer: Any = state.get("formatted_response") if succeeded else ""
        return {
            "output": answer or "",
            "status": status,
            "disclaimer_applied": bool(state.get("disclaimer_applied", False)),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
