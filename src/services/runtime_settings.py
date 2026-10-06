"""AgentCore Platform v1.0"""

# Runtime settings resolved from config/config.yaml.
#
# Every tunable the pipeline reads is resolved HERE, once, through a
# finite-and-bounded parser. Two properties matter:
#
#   1. A declared value must actually reach the code that uses it. Runtime
#      parameters live in config/config.yaml and are handed to the graph as
#      Graph(config=...); the graph forwards them to the inner workflow and
#      seeds the resolved settings into state so every node reads the same
#      values.
#   2. A malformed value must FAIL, not degrade. `float()` happily parses
#      "NaN" and "Infinity", and every comparison against NaN is False — so an
#      unchecked bound silently stops bounding anything. `_finite_in_range`
#      rejects those, along with bools, non-numerics and out-of-range
#      magnitudes, and names the offending field without echoing its value.

from __future__ import annotations

import math
from typing import Any, Mapping

# State key carrying the resolved settings for the duration of one invocation.
RUNTIME_SETTINGS_KEY = "runtime_settings"


class SettingsError(ValueError):
    """A declared runtime setting is missing, malformed or out of bounds."""


# field -> (config path, default, minimum, maximum)
_INT_SETTINGS: dict[str, tuple[tuple[str, ...], int, int, int]] = {
    # Hard ceiling on the caller's clinical question.
    "query_max_chars": (("query", "max_chars"), 2000, 1, 20000),
    # Number of knowledge-base passages retrieved for one question.
    "retrieval_top_k": (("retrieval", "top_k"), 3, 1, 20),
    # How much of the question is echoed into a retrieved passage header.
    "retrieval_query_excerpt_chars": (("retrieval", "query_excerpt_chars"), 80, 8, 2000),
    # How much of the question is echoed into the rendered answer header.
    "answer_query_excerpt_chars": (("retrieval", "answer_excerpt_chars"), 120, 8, 2000),
    # Rendering caps for the assembled clinical answer.
    "max_guidelines": (("clinical", "max_guidelines"), 2, 0, 20),
    "max_icd11_codes": (("clinical", "max_icd11_codes"), 5, 0, 50),
    # Bound on the caller-supplied clinical-term channel.
    "max_context_terms": (("clinical", "max_context_terms"), 8, 0, 32),
}


def _finite_in_range(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    integer: bool = True,
) -> float:
    """Parse *value* as a finite number inside [minimum, maximum] or raise.

    Rejects, in order: booleans (``isinstance(True, int)`` is True in Python, so
    ``enabled: true`` would otherwise pass as ``1``), non-numeric types and
    unparsable strings, NaN / +-Infinity, non-integers when *integer* is set, and
    out-of-range magnitudes.

    The raised message names the FIELD only — never the rejected value, which is
    caller- or operator-controlled text that must not be echoed onward.
    """
    if isinstance(value, bool):
        raise SettingsError(f"{field}: must be a number, not a boolean")
    if isinstance(value, str):
        try:
            parsed: float = float(value.strip())
        except (TypeError, ValueError):
            raise SettingsError(f"{field}: not a number") from None
    elif isinstance(value, (int, float)):
        parsed = float(value)
    else:
        raise SettingsError(f"{field}: not a number")

    if not math.isfinite(parsed):
        raise SettingsError(f"{field}: must be finite")
    if integer and parsed != int(parsed):
        raise SettingsError(f"{field}: must be a whole number")
    if not (minimum <= parsed <= maximum):
        raise SettingsError(f"{field}: outside the permitted range [{minimum}, {maximum}]")
    return parsed


# Distinguishes "the key is absent" from "the key was declared with no value".
# A key written with an empty value is a declaration error, not a request for
# the default: someone meant to put a number there.
_MISSING = object()


def _dig(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Return the value at *path* in a nested mapping, or _MISSING when absent.

    A non-mapping at an intermediate step is a declaration error, not a miss:
    ``retrieval: 3`` must fail loudly rather than silently fall back to defaults.
    """
    node: Any = config
    for i, key in enumerate(path):
        if not isinstance(node, Mapping):
            raise SettingsError(f"{'.'.join(path[:i])}: must be a mapping")
        if key not in node:
            return _MISSING
        node = node[key]
    return node


def resolve_runtime_settings(config: Mapping[str, Any] | None) -> dict[str, int]:
    """Resolve every runtime setting from *config*, failing closed on bad values.

    Absent keys take the documented default; present keys are validated. The
    return value is a flat dict of primitives so it can be carried in graph
    state (checkpoint serialisation admits primitives only).
    """
    source: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    resolved: dict[str, int] = {}
    for field, (path, default, minimum, maximum) in _INT_SETTINGS.items():
        raw = _dig(source, path)
        if raw is _MISSING:
            resolved[field] = default
            continue
        resolved[field] = int(
            _finite_in_range(
                raw,
                field=".".join(path),
                minimum=minimum,
                maximum=maximum,
            )
        )
    return resolved


def settings_from_state(state: Mapping[str, Any]) -> dict[str, int]:
    """Read the resolved settings a graph seeded into *state*.

    Falls back to the documented defaults when the key is absent, so a node
    called directly in a test behaves the same as one called through a graph.
    """
    seeded = state.get(RUNTIME_SETTINGS_KEY)
    if isinstance(seeded, Mapping):
        return {field: int(seeded.get(field, default)) for field, (_path, default, _lo, _hi) in _INT_SETTINGS.items()}
    return resolve_runtime_settings({})
