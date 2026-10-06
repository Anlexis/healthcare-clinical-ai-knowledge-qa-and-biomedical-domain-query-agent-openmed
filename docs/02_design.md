# Template Design Specification — HCR-C2-052 ClinicalBiomedicalQAAgent

## Position in the framework

- **Agent class**: `ClinicalBiomedicalQAAgent`
- **Layer separation**
  - State: flat typed dict (no model objects — checkpoint serialisation corrupts them)
  - Node: framework inheritance, overriding `execute(self, state) -> dict` only
  - Graph: composition via `register_nodes()`

| Layer | Base |
|---|---|
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner workflow base | `BaseGraph` — direct framework inheritance |

## Architecture overview

### Pattern

A two-layer domain pipeline. The outer graph runs the fixed five-node backbone
(initialize → pre_process → main → post_process → finalize). The `main` slot holds a subgraph
node (`BiomedicalQAGraphNode`) wrapping `BiomedicalDomainWorkflowGraph`, which contains the
six-step clinical Q&A pipeline.

```
Outer backbone (fixed):
START → initialize → pre_process → [BiomedicalQAGraphNode] → post_process → finalize → END
                                            |
                    (inner domain workflow) v
  query_parse → domain_route → biobert_retrieve → clinical_kb_lookup → response_format → output_gate
```

### Node configuration

| Node | Layer | Responsibility | Trust level |
|------|-------|----------------|-------------|
| initialize | Backbone | Session bootstrap, schema version | ANONYMOUS |
| pre_process | Backbone | Caller-data contract: trust, question, structured context | VERIFIED_EXTERNAL |
| main (BiomedicalQAGraphNode) | Backbone / bridge | Wraps the inner domain workflow | ANONYMOUS |
| post_process | Backbone | Publication boundary — enforces the published invariants | ANONYMOUS |
| finalize | Backbone | Response metadata, timings | ANONYMOUS |
| query_parse | Inner | Intent classification, code-candidate extraction, re-screen | ANONYMOUS |
| domain_route | Inner | Route to the biomedical knowledge domain | ANONYMOUS |
| biobert_retrieve | Inner | Retrieve passages from the on-premise index | ANONYMOUS |
| clinical_kb_lookup | Inner | Published guidance + classification-code candidates | ANONYMOUS |
| response_format | Inner | Assemble the cited answer | ANONYMOUS |
| output_gate | Inner | Attach the mandatory clinical disclaimer | ANONYMOUS |

### Data flow

```
user_input + input_context
    |
[PreProcessNode]        -> validated_input, validated_context, input_redacted, enriched_context
    |
[BiomedicalQAGraphNode] (subgraph node)
    |  extract_input() publishes the validated outer fields across the boundary
    |
    [QueryParseNode]         -> query_intent, extracted_entities
    [DomainRouteNode]        -> domain_routed
    [BioBERTRetrieveNode]    -> retrieved_passages          (on-premise index)
    [ClinicalKBLookupNode]   -> guideline_matches, icd11_codes
    [ResponseFormatNode]     -> clinical_response           (cited answer)
    [OutputGateNode]         -> formatted_response, disclaimer_applied
    |
merge_output: result <- inner formatted_response (success only), disclaimer_applied
    |
[PostProcessNode]       -> result, formatted_output        (publication boundary)
    |
[FinalizeNode]          -> response_metadata, total_time_ms
```

### State definition

| Field | Type | Set by | Purpose |
|-------|------|--------|---------|
| runtime_settings | dict | both graphs (initial state) | Resolved, validated runtime parameters |
| validated_input | str | PreProcessNode | Screened question |
| validated_context | dict | PreProcessNode | Screened structured caller context |
| input_redacted | bool | PreProcessNode | The platform filter rewrote the question |
| enriched_context | dict | PreProcessNode | Channel and source metadata |
| query_intent | str | QueryParseNode | Detected clinical intent |
| extracted_entities | list | QueryParseNode | Classification-code candidates |
| domain_routed | str | DomainRouteNode | Selected knowledge domain |
| retrieved_passages | list | BioBERTRetrieveNode | Passages from the on-premise index |
| guideline_matches | list | ClinicalKBLookupNode | Matching published guidance |
| icd11_codes | list | ClinicalKBLookupNode | Unverified code candidates |
| clinical_response | str | ResponseFormatNode | Cited answer body |
| formatted_response | str | OutputGateNode | Answer with the mandatory disclaimer |
| disclaimer_applied | bool | OutputGateNode, then merge_output | Compliance provenance |
| result | str | merge_output, then PostProcessNode | Answer carried to the boundary |
| formatted_output | str | PostProcessNode | Published payload, or the withheld notice |

**State constraints (mandatory)**

- Flat typed dict only (primitives and JSON-serialisable types)
- No credentials, tokens or keys in state — checkpoints are persisted
- Invocation context travels in the runnable config, never in state
- No model objects, dataclasses or arbitrary Python objects

## Runtime configuration

`config/agent.yaml` is the static deployment manifest. Runtime parameters live in
`config/config.yaml` and are handed to the graph as `Graph(config=...)`:

| Key | Read by |
|---|---|
| `query.max_chars` | PreProcessNode |
| `retrieval.top_k` | BioBERTRetrieveNode, ResponseFormatNode |
| `retrieval.query_excerpt_chars` | BioBERTRetrieveNode |
| `retrieval.answer_excerpt_chars` | ResponseFormatNode |
| `clinical.max_guidelines`, `clinical.max_icd11_codes` | ResponseFormatNode |
| `clinical.max_context_terms` | PreProcessNode |

Every value is resolved once, at start-up, by `src/services/runtime_settings.py` through a
finite-and-bounded parser, and seeded into invocation state so the outer backbone and the inner
workflow read identical values. A malformed or out-of-range value fails the compile; it never
degrades to a default. NaN and the infinities are rejected explicitly — they parse through
`float()` and then compare False against every bound, which silently disables the bound.

## Boundary contracts

### Caller-data contract

`POST /invoke` accepts the question plus an optional structured context object with two declared
fields, `channel` and `clinical_terms`. Values are restricted to a closed identifier alphabet
(`[a-z0-9_-]`, 1–32 characters) because they are rendered into the answer, and the entry point
DROPS undeclared keys rather than ignoring them: an ignored key still reaches graph state, where
the first backbone node returns it verbatim into its result and the platform output gate scans it.

The structured channel exists because the platform input filter masks personal-data shapes inside
the free-text question, and the heuristic personal-name pattern it uses also matches ordinary
capitalised clinical terminology — "Type 2 Diabetes Mellitus" is rewritten to "Type 2 [MASKED]".
Terms supplied on the structured channel survive intact. When the filter has rewritten the
question, PreProcessNode records that, and the answer carries an explicit notice rather than
reading as a complete reply to a question the reader did not ask.

### Boundary bridge

The subgraph node contract forwards only the input string to the inner graph, so an inner node
reading an outer state key sees the inner graph's own state, where that key does not exist. The
validated context and the redaction flag therefore travel over `src/graph/context_bridge.py`:
`extract_input()` publishes them immediately before the inner invocation, and the inner graph's
`_extra_initial_state()` seeds them. The publish is unconditional, so nothing survives from one
invocation into the next.

### Publication boundary

`PostProcessNode` is the single place the published invariants are enforced, and it checks the
value actually returned to the caller rather than one it assembled itself:

1. the mandatory clinical disclaimer is present;
2. no protected identifier appears in the answer;
3. no credential shape appears in the answer.

Check 2 is re-run HERE, after input validation, because an identifier can enter the answer later:
from retrieved knowledge-base content, or by COMPOSING caller terms that are each inert on their
own. Rendering `["pt", "98765"]` as a term line produces `pt 98765`, a record-number shape that no
per-field input check can see. That composition hazard is owned by this boundary by design.

On a violation the answer-bearing fields are CLEARED and a non-empty notice is published in their
place. An error status alone withholds nothing: the caller-visible payload resolves as the
formatted output OR the raw result with no status check, so a refusal that leaves the result field
populated ships the ungated answer inside the error envelope, and an empty notice falls through to
the same place. Violation messages carry a reason code and a field name only — a credential-shaped
string anywhere in the returned mapping makes the platform output gate raise, and the node wrapper
then discards the whole mapping, clearing included.

The inner `OutputGateNode` APPLIES the disclaimer; it does not certify it. A check placed there
could only re-read the string the same method had just built, which cannot fail. Only one layer
enforces each invariant, deliberately: a second, redundant layer would contain the violation on
its own and make the layer under test impossible to falsify.

### Screening

`src/services/screening.py` holds three screens, each returning a stable reason code and never the
matched text:

| Screen | Basis |
|---|---|
| Protected identifiers | Domain patterns for record / chart / insurance numbers, plus the platform PII detector as a superset for structured forms (email, phone, national ID, payment card, labelled names) |
| Prompt injection | Directive phrases, bilingual, plus chat-template control tokens (`<\|…\|>`, `[INST]`, `<<SYS>>`) screened as a class |
| Credentials | Delegated wholesale to the platform detector |

Every screen runs on BOTH the original text and its compatibility-normalised form. Full-width
punctuation and digits are the ordinary way to type Japanese clinical text, so a half-width-only
pattern silently passes `患者番号：12345678`; and scanning only the normalised copy would miss
what normalisation removes, while the answer renders the original.

The platform detector's heuristic personal-name type is excluded from the identifier screen, and
only that type. It matches any run of two or more capitalised words, which in this domain is the
normal shape of a condition or an active ingredient, so treating it as a rejection would refuse
ordinary clinical questions. The anchored labelled-name type is kept.

The credential screen delegates rather than duplicating: a local pattern set narrower than the
platform's is a bypass, because the value passes the template's check, the platform gate then
raises inside the node wrapper, and the wrapper discards the node's whole delta — including any
clearing performed.

## Directory layout

```
src/
  api/
    server.py                   <- entry-point auth + context normalisation
  graph/
    graph.py                    <- outer agent graph and the main-slot subgraph node
    domain_workflow_graph.py    <- inner workflow graph
    context_bridge.py           <- outer -> inner boundary transport
  nodes/
    pre_process_node.py         <- caller-data contract (VERIFIED_EXTERNAL)
    post_process_node.py        <- publication boundary (ANONYMOUS)
    query_parse_node.py         <- inner domain node (ANONYMOUS)
    domain_route_node.py        <- inner domain node (ANONYMOUS)
    biobert_retrieve_node.py    <- inner domain node (ANONYMOUS)
    clinical_kb_lookup_node.py  <- inner domain node (ANONYMOUS)
    response_format_node.py     <- inner domain node (ANONYMOUS)
    output_gate_node.py         <- inner domain node (ANONYMOUS)
  schemas/
    state.py                    <- flat typed dict state
  services/
    runtime_settings.py         <- validated runtime parameters
    screening.py                <- identifier / injection / credential screens
    caller_contract.py          <- structured context contract
```

## Framework utilisation

- [x] Invocation context (correlation id, session id, trust level, caller id)
- [x] Trust gate: `required_trust_level = TrustLevel.VERIFIED_EXTERNAL` on PreProcessNode,
      matching the manifest's declared entry level and the trust the entry point grants
- [x] Input screening in the node that owns the caller contract, proved by calling `execute()`
      directly rather than relying on a platform gate being active
- [x] Output invariants enforced at the publication boundary, with clearing on violation
- [x] `emit_trace_event()` — a domain audit event on every node's execution path
- [x] No credential fields in state; the knowledge index is reached inside the deployment network

## Composition

- **Pattern**: subgraph node wrapping an inner workflow graph
- **Error propagation**: propagate — an inner failure surfaces as a subgraph error
- **Interrupt propagation**: disabled (`propagate_hitl = False`)

## Deployment notes

- **On-premise**: knowledge-base retrieval must not leave the deployment network
- **Mandatory disclaimer**: every published answer contains
  `この回答は参考情報です。最終判断は医師にご確認ください。`
- **Classification codes** are reported as unverified candidates extracted from the question. The
  extractor matches a code SHAPE, which any abbreviation in a clinical question also has, so
  labelling them as verified register entries would put a provenance claim on the answer that
  nothing in the pipeline established.

## Import isolation

- [x] Template code does not import platform internals
- [x] Import targets: the framework and shared packages only
- [x] No `.run()` calls — only `.invoke()`

## Design decision record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base graph | `AgentBaseGraph` | `AutonomousBaseGraph` | `AgentBaseGraph` | Fixed pipeline; no autonomous loop |
| Composition | Flat backbone slots | Nested subgraph | Nested subgraph | Six domain steps |
| Inner graph base | `BaseGraph` | `AgentBaseGraph` | `BaseGraph` | Fully custom topology |
| Invariant enforcement | Inner output node | Publication boundary | Publication boundary | It is the value the caller receives, and it is falsifiable |
| Redaction handling | Refuse the request | Answer with an explicit notice | Notice | The filter fires on ordinary clinical terminology; refusing would block real work |
| Composed-identifier hazard | Screen the render in place | Catch it at the boundary | Boundary | One enforcing layer keeps it falsifiable |
