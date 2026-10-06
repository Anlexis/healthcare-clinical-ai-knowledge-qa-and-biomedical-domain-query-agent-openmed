"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a validating model object.
# Checkpoints are serialised with msgpack, and model instances corrupt
# silently. Extend the framework state with agent-specific fields only.
# Never add credentials or secrets.

from typing import Any, Optional

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """ClinicalBiomedicalQAAgent state.

    Field write chain:
      PreProcessNode       -> validated_input, validated_context,
                              input_redacted, enriched_context
      QueryParseNode       -> query_intent, extracted_entities
      DomainRouteNode      -> domain_routed
      BioBERTRetrieveNode  -> retrieved_passages
      ClinicalKBLookupNode -> guideline_matches, icd11_codes
      ResponseFormatNode   -> clinical_response
      OutputGateNode       -> formatted_response, disclaimer_applied
      main-slot node       -> result, disclaimer_applied (via merge_output)
      PostProcessNode      -> result, formatted_output

    Both graphs seed runtime_settings before the first node runs.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from the framework state.
    """

    # Resolved runtime settings, seeded by the graph for the whole invocation
    runtime_settings: Optional[dict[str, Any]]

    # Backbone fields (PreProcessNode)
    validated_input: Optional[str]
    validated_context: Optional[dict[str, Any]]
    input_redacted: Optional[bool]
    enriched_context: Optional[dict[str, Any]]

    # Inner domain fields (QueryParseNode)
    query_intent: Optional[str]
    extracted_entities: Optional[list[Any]]

    # Inner domain fields (DomainRouteNode)
    domain_routed: Optional[str]

    # Inner domain fields (BioBERTRetrieveNode)
    retrieved_passages: Optional[list[Any]]

    # Inner domain fields (ClinicalKBLookupNode)
    guideline_matches: Optional[list[Any]]
    icd11_codes: Optional[list[Any]]

    # Inner domain fields (ResponseFormatNode)
    clinical_response: Optional[str]

    # Inner domain fields (OutputGateNode)
    formatted_response: Optional[str]
    disclaimer_applied: Optional[bool]

    # Outer merge (main-slot subgraph node)
    result: Optional[str]

    # Backbone output (PostProcessNode)
    formatted_output: Optional[str]
