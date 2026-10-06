# Test Specification — HCR-C2-052 ClinicalBiomedicalQAAgent

## Overview

The suite covers four things:

1. unit tests for every backbone and domain node, plus the three service modules;
2. input-boundary tests, asserted by calling `execute()` DIRECTLY so a rejection proves this
   template refuses the payload rather than that a platform gate happened to be active;
3. publication-boundary tests, asserting presence AND emptiness on every cleared field;
4. end-to-end tests through the real HTTP entry point, auth included.

## Test matrix

### Unit tests — nodes (`tests/unit/test_agent.py`)

| Test class | Scenarios | Expected |
|------------|-----------|----------|
| TestPreProcessNode | empty, whitespace-only, valid question, injection, control token, four identifier forms, length bound from config, valid / invalid structured context, platform-redaction flag | ERROR with nothing carried forward, or SUCCESS with the validated fields |
| TestQueryParseNode | three identifier forms, five intent classifications, code-candidate labelling, legitimate query | ERROR, or SUCCESS with the classified intent |
| TestDomainRouteNode | six intent-to-domain mappings including an unknown intent | SUCCESS with the routed domain |
| TestBioBERTRetrieveNode | passage fields, `retrieval.top_k` from config, caller terms retrieved as their own queries | SUCCESS with a bounded passage list |
| TestClinicalKBLookupNode | guidance for guideline intents, none otherwise, candidates never marked verified | SUCCESS with unverified candidates |
| TestResponseFormatNode | passages and guidance rendered, redaction notice present / absent, candidates rendered as unverified, rendering caps from config | SUCCESS with the assembled answer |
| TestOutputGateNode | disclaimer appended, present on an empty body, never duplicated | SUCCESS with the disclaimer |

### Unit tests — services

| File | Coverage |
|---|---|
| `tests/unit/test_runtime_settings.py` | Defaults; declared values reaching the resolver; a per-field non-finite matrix (`"NaN"`, `"Infinity"`, `"-Infinity"`, raw NaN / ±inf, over-magnitude, fractional, non-numeric, boolean, declared-but-empty); a block declared as a scalar; error messages naming the field and not the value; state seeding |
| `tests/unit/test_screening.py` | A clean clinical corpus that must NOT trip any screen; sixteen identifier forms in half-width and full-width; eleven injection attempts including control tokens; credential parity asserted as an identity against the platform detector; findings never carrying the matched text |
| `tests/unit/test_caller_contract.py` | Accepted shapes; unknown field refused; unsafe field name reported by position; free-text and over-length terms refused; term cap; credential / identifier / injection screens on the channel; refusal set pinned to the platform detector; undeclared keys dropped |

### Framework compliance (`tests/unit/test_framework_compliance_tc06_tc07.py`)

The default input and output gates cannot be replaced by a domain node — both raise at class
definition time.

### Boundary tests

#### Input boundary (`tests/proof_of_boundary/test_pb_input_boundary.py`)

| Test group | Coverage |
|---|---|
| Trust declarations | The entry node's required level equals the manifest's declared level; no inner node raises the bar above it |
| Protected identifiers | Seven identifier forms refused; four legitimate clinical questions — including capitalised condition names — still pass |
| Injection | Five attack forms refused, including chat-template control tokens; defence in depth at the inner parse node |
| Structured context | Six hostile context shapes refused; an ordinary one passes |
| Numeric bounds | Six non-finite / out-of-range configuration values refused at compile |

#### Publication boundary (`tests/proof_of_boundary/test_pb_output_boundary.py`)

| Test group | Coverage |
|---|---|
| Clean-path control | A compliant answer is published unchanged, and the withheld notice is non-empty |
| Withholding paths | Missing disclaimer; identifier in the answer; identifier COMPOSED from individually-inert caller terms; credential shape; empty answer |
| Refusal messages | Reason code and field name only; no matched text, no traceback, no source path |
| Field inventory | The declared answer-bearing fields are exactly those cleared, each present in the returned update |

Every clearing assertion checks presence AND emptiness. A partial update that omits a key leaves
the previous value in state, so `not result.get(field)` would pass on a gate that clears nothing.

#### Backbone order (`tests/proof_of_boundary/test_pb_invoke_order.py`)

| Test | Expected |
|---|---|
| `test_backbone_invoke_order_success` | SUCCESS, node history equals the five backbone nodes, disclaimer present |
| `test_external_caller_context_is_the_one_exercised` | The verified-external level is the one used — never an internal context, which would hide an over-demanding node |
| `test_an_anonymous_caller_is_refused_and_publishes_nothing` | ERROR, empty output, publication boundary never reached |

#### End-to-end through the entry point (`tests/proof_of_boundary/test_pb_invoke_endpoint.py`)

| Test group | Coverage |
|---|---|
| Entry-point auth | Health open; unauthenticated and wrong-token callers refused with 401; an authenticated caller reaches the pipeline at the manifest's declared level |
| Published answer | Non-empty answer carrying the disclaimer; backbone order; the stored deployment payload is answered |
| Caller data reaching the inner workflow | Structured terms appear in the answer; the answer is computed from them; undeclared keys are dropped rather than carried |
| Rejections | Credential-shaped context value refused with 400 naming the field and not the value; ordinary domain text on the same field still passes; ten refused payloads publish nothing; an identifier is never echoed back |
| Containment | A composed identifier is withheld; the error envelope carries no released text, no traceback and no source path; the notice is non-empty; the block happened at the publication boundary; a clean-path control still produces the real answer |
| Declared configuration | `retrieval.top_k` of 1 and 3 produce 1 and 3 rendered passages; the shipped configuration file is the one the entry point loads |

#### Interrupt propagation (`tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py`)

Skip stub — this template does not enable cross-boundary interrupt propagation, so there is no
propagation behaviour to assert. The module is real and importable, and it skips with a reason.

#### Structural boundary tests

| File | Coverage |
|---|---|
| `tests/proof_of_boundary/test_import_isolation.py` | Template code does not import platform internals |
| `tests/proof_of_boundary/test_state_safety.py` | State carries no credential-like field and no prohibited type |

## Security coverage

| Concern | Test file | Coverage |
|---|---|---|
| Trust boundary | test_pb_input_boundary.py, test_pb_invoke_endpoint.py | Declared level matches the manifest AND the trust the entry point grants; an anonymous caller is refused |
| Protected identifiers, input | test_pb_input_boundary.py, test_screening.py | Half-width and full-width forms refused; legitimate clinical terminology unaffected |
| Protected identifiers, output | test_pb_output_boundary.py, test_pb_invoke_endpoint.py | The finished answer is re-scanned; composed identifiers withheld |
| Injection | test_pb_input_boundary.py, test_screening.py | Directive phrases and chat-template control tokens |
| Credentials | test_caller_contract.py, test_screening.py, test_pb_invoke_endpoint.py | Refusal set pinned by identity to the platform detector; context values refused at the entry point |
| Containment | test_pb_output_boundary.py, test_pb_invoke_endpoint.py | Answer-bearing fields cleared; non-empty notice; no released text in the error envelope |
| Audit | test_agent.py (all nodes) | Audit emission patched per node module rather than stubbed wholesale, so a missing call is visible |

## Valid test payload

```python
VALID_PAYLOAD = "What are the drug-drug interactions for warfarin and aspirin per MHLW guidelines?"
```

Non-empty, within the configured length bound, carries no protected identifier, and classifies as
a drug-interaction question — so it reaches a successful answer through the full backbone. It is
the same payload stored in `deploy/invoke_payload.json`'s sibling role for deployment sign-off,
and the suite asserts that stored payload is one this agent actually serves.
