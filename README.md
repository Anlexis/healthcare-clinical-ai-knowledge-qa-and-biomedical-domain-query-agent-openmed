# Clinical & Biomedical Knowledge Q&A Agent

AI agent for answering clinical and biomedical domain questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Healthcare
> **Template ID**: HCR-C2-052

## Overview

A clinical knowledge question-answering agent for healthcare teams.

Given a clinical question — a drug-interaction query, a treatment-guideline lookup, a
classification-code question — it routes the question to the matching biomedical knowledge
domain, retrieves supporting passages from a knowledge base deployed inside your own network,
cross-references published clinical guidance, and returns a cited answer carrying a mandatory
"reference information only, confirm with a physician" disclaimer.

Two properties make it suitable for clinical use rather than general question answering:

- **Nothing leaves your network.** Retrieval runs against a locally deployed index; the pipeline
  makes no outbound call and invokes no language model.
- **Patient identifiers are refused, not answered.** Questions carrying a record number, chart
  number, insurance number or other personal identifier are rejected at the entry point, in both
  half-width and full-width Japanese forms. The finished answer is re-scanned before publication,
  because an identifier can also arrive from retrieved content or be formed by composing
  individually-harmless caller terms. A failed re-scan withholds the answer instead of returning
  it with a warning attached.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. The agent imports its base classes and security gates from the framework package at
start-up, so without that package installed and configured, import and graph compile fail
outright rather than leaving the agent partially working. This is intentional — a half-running
clinical agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Configuration

`config/agent.yaml` is the deployment manifest — identity, entry class and the trust level a
caller must hold. `config/config.yaml` holds the runtime parameters, and every value in it is
validated at start-up: a malformed or out-of-range entry fails the compile rather than quietly
falling back to a default.

| Key | Meaning |
|---|---|
| `query.max_chars` | Hard ceiling on the caller's question |
| `retrieval.top_k` | Knowledge-base passages retrieved and rendered per question |
| `retrieval.query_excerpt_chars` | Question characters carried into a passage header |
| `retrieval.answer_excerpt_chars` | Question characters echoed in the answer header |
| `clinical.max_guidelines` | Guidance entries rendered |
| `clinical.max_icd11_codes` | Classification-code candidates rendered |
| `clinical.max_context_terms` | Structured clinical terms accepted per request |

## Calling the agent

`POST /invoke` takes the question plus an optional structured context object:

```json
{
  "input": "What are the drug-drug interactions for warfarin and aspirin?",
  "input_context": {"channel": "portal", "clinical_terms": ["warfarin", "aspirin"]}
}
```

`clinical_terms` exists because the platform's input filter masks personal-data shapes inside
free text, and the heuristic personal-name pattern it uses also matches ordinary capitalised
clinical terminology such as "Type 2 Diabetes Mellitus". Terms supplied on the structured channel
survive intact. They are restricted to a closed identifier alphabet (`[a-z0-9_-]`, up to 32
characters) precisely because they are rendered into the answer, and undeclared keys are dropped
at the entry point rather than carried into the pipeline. When the input filter has rewritten the
question, the answer says so instead of presenting a complete-looking reply.

## Customising

1. Adjust `config/config.yaml` for your own environment and policies.
2. Point `BioBERTRetrieveNode` at your own knowledge index, and `ClinicalKBLookupNode` at your own
   guidance and code registers. The shipped implementations return representative results so the
   pipeline is runnable before either is connected.
3. Extend the identifier and injection patterns in `src/services/screening.py` for your own
   record-number conventions.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

---
