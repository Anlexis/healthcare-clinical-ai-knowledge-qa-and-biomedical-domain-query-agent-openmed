"""AgentCore Platform v1.0"""

# The caller-data contract for the structured context channel.
#
# A request carries two things: the clinical question (free text) and an
# optional structured context object. The structured channel exists because the
# platform input filter masks personal-data shapes inside the free-text
# question, and the heuristic personal-name pattern it uses matches ordinary
# capitalised clinical terminology ("Type 2 Diabetes Mellitus"). A caller that
# needs a specific condition or active ingredient to survive intact supplies it
# here instead, in an INERT form.
#
# "Inert" is the whole safety argument for that channel, so it is enforced
# rather than assumed:
#
#   * every value is matched against a closed identifier alphabet — free text
#     rendered into an answer is caller-controlled output;
#   * the number of entries is capped from configuration;
#   * unknown keys are refused (validators that merely IGNORE an unknown key
#     leave it in state, where it is returned verbatim by the first backbone
#     node and scanned by the platform output gate — one credential-shaped
#     value there fails the request with an opaque error the caller cannot act
#     on);
#   * key NAMES are caller data too, so an unrecognised name is reported by
#     position unless the name itself is inert;
#   * values are screened for credential shapes and protected identifiers,
#     because the platform's own input filter does not scan this channel.

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from src.services.screening import (
    contains_credentials,
    scan_for_injection,
    scan_for_protected_identifiers,
)

# Closed alphabet for anything the caller can push into the rendered answer.
_INERT_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

# A key name is only quoted back to the caller when it is itself inert.
_SAFE_KEY_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

CONTEXT_FIELDS = ("channel", "clinical_terms")


def _name_for_message(key: Any, position: int) -> str:
    if isinstance(key, str) and _SAFE_KEY_NAME.match(key):
        return f"input_context.{key}"
    return f"input_context field #{position}"


# The three screen messages name the detector that caught the value. A caller given those
# back can probe the screen one payload at a time, so they are withheld -- unlike the
# inert-form message, which names a rule the caller needs in order to fix the request.
_SCREEN_MARKERS = (
    "credential-shaped value refused",
    "protected identifier refused",
    "instruction-injection pattern refused",
)


def is_screen_rejection(reason: str) -> bool:
    """True when the reason names the screen that fired rather than a field rule."""
    return any(marker in reason for marker in _SCREEN_MARKERS)


def _screen_string(value: str, field: str) -> Optional[str]:
    """Return a rejection reason for *value*, or None when it is acceptable."""
    if contains_credentials(value):
        return f"{field}: credential-shaped value refused"
    code = scan_for_protected_identifiers(value)
    if code is not None:
        return f"{field}: protected identifier refused ({code})"
    code = scan_for_injection(value)
    if code is not None:
        return f"{field}: instruction-injection pattern refused ({code})"
    if not _INERT_IDENTIFIER.match(value):
        return f"{field}: must match the inert identifier form [a-z0-9_-], 1-32 characters"
    return None


def validate_caller_context(
    raw: Any,
    *,
    max_terms: int,
) -> tuple[dict[str, Any], Optional[str]]:
    """Validate the structured context channel.

    Returns ``(validated, error)``. On rejection *validated* is empty and
    *error* names the offending field — never its value, and never the matched
    span of a screening finding.

    An absent or empty channel is valid: the pipeline then answers from the
    free-text question alone.
    """
    if raw is None:
        return {}, None
    if not isinstance(raw, Mapping):
        return {}, "input_context: must be an object"

    validated: dict[str, Any] = {}
    for position, (key, value) in enumerate(raw.items(), start=1):
        name = _name_for_message(key, position)
        if not isinstance(key, str) or key not in CONTEXT_FIELDS:
            return {}, f"{name}: unrecognised field"

        if key == "channel":
            if not isinstance(value, str):
                return {}, f"{name}: must be a string"
            reason = _screen_string(value, name)
            if reason is not None:
                return {}, reason
            validated["channel"] = value
            continue

        # clinical_terms
        if not isinstance(value, list):
            return {}, f"{name}: must be a list"
        if len(value) > max_terms:
            return {}, f"{name}: at most {max_terms} entries"
        terms: list[str] = []
        for index, item in enumerate(value, start=1):
            entry = f"{name}[{index}]"
            if not isinstance(item, str):
                return {}, f"{entry}: must be a string"
            reason = _screen_string(item, entry)
            if reason is not None:
                return {}, reason
            terms.append(item)
        validated["clinical_terms"] = terms

    return validated, None


def drop_unknown_context_fields(raw: Any) -> dict[str, Any]:
    """Keep only the declared context fields, discarding everything else.

    Used at the HTTP boundary. Dropping — rather than ignoring — is what makes
    the channel structurally safe: an undeclared key that merely fails
    validation later has already been carried into graph state by then.
    """
    if not isinstance(raw, Mapping):
        return {}
    return {key: raw[key] for key in CONTEXT_FIELDS if key in raw}
