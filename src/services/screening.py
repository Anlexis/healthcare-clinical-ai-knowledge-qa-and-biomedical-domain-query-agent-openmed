"""AgentCore Platform v1.0"""

# Content screening for the clinical Q&A pipeline.
#
# Three independent screens, each with its own reason code:
#
#   protected identifiers  — patient/chart/insurance numbers and the structured
#                            personal identifiers the framework detector knows.
#   prompt injection       — directive phrases AND chat-template control tokens.
#   credentials            — delegated wholesale to the framework detector.
#
# Two rules shape the implementation:
#
#   Screen the text in BOTH the form it arrived in and the form it normalises
#   to. Half-width and full-width punctuation are interchangeable in Japanese
#   clinical text, so a pattern written against one form silently misses the
#   other; and a screen that ran only on a normalised copy would miss anything
#   normalisation removes. Both directions are scanned, because it is the
#   ORIGINAL string that gets rendered into the answer.
#
#   A finding names a stable reason CODE, never the matched text. Screening
#   output travels into audit events and error messages, and the whole point of
#   the screen is that the matched span must not be copied anywhere.

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

from framework.security.credential_detector import detect_credentials_in_value
from framework.security.pii_detector import detect_pii

# ── Protected clinical identifiers ────────────────────────────────────────────
# Labels seen on Japanese clinical records. The separator class admits the
# half-width and full-width colon, the ideographic space, and the long-vowel /
# dash forms that appear in hand-typed record numbers.
_JP_ID_LABEL = (
    r"(?:患者番号|患者ID|患者Ｉ?Ｄ|患者No|患者コード|カルテ番号|カルテNo|" r"診察券番号|被保険者番号|受診番号)"
)
_ID_SEP = r"[\s　]*[:：\-ー－.．]?[\s　]*"

_PROTECTED_ID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Labelled record numbers — the dominant form in Japanese clinical text.
    ("protected_id.record_number", re.compile(_JP_ID_LABEL + _ID_SEP + r"[0-9０-９]{2,}")),
    # Bare long digit runs: patient identifiers are 10-12 digits in this domain.
    ("protected_id.numeric", re.compile(r"(?<![0-9０-９])[0-9０-９]{10,12}(?![0-9０-９])")),
    # Western record-number forms.
    ("protected_id.record_prefix", re.compile(r"\b(?:PT|MRN)[-/\s]?\d{4,}\b", re.IGNORECASE)),
]

# The framework's own PII detector is used as a SUPERSET for the structured
# identifier forms (email, phone, national ID, payment card, labelled Japanese
# names) so this template can never be narrower than the platform on those.
#
# Its heuristic `name` type is deliberately excluded. That pattern matches any
# run of two or more capitalised words, which in this domain is the normal shape
# of a condition or an active ingredient — "Type 2 Diabetes Mellitus",
# "Warfarin Sodium", "Alzheimer Disease". Treating those as personal
# identifiers would refuse ordinary clinical questions, and the detector's own
# contract invites callers working over structured text to post-filter that
# type. The anchored `name_jp` type, which fires only after an explicit name
# label, is kept.
_HEURISTIC_PII_TYPES = frozenset({"name"})

# ── Prompt injection ──────────────────────────────────────────────────────────
# Directive phrases, bilingual. Anchored on the verb so ordinary clinical prose
# that happens to contain one of these words is unaffected.
_INJECTION_PHRASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "injection.instruction_override",
        re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions?", re.IGNORECASE),
    ),
    (
        "injection.instruction_override",
        re.compile(
            r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompt)",
            re.IGNORECASE,
        ),
    ),
    (
        "injection.instruction_override",
        re.compile(
            r"forget\s+(?:all\s+|everything\s+)?(?:your\s+|the\s+)?(?:previous\s+|prior\s+)?"
            r"(?:instructions?|rules?|prompt)",
            re.IGNORECASE,
        ),
    ),
    ("injection.prompt_disclosure", re.compile(r"(?:system|developer)\s+prompt", re.IGNORECASE)),
    (
        "injection.prompt_disclosure",
        re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", re.IGNORECASE),
    ),
    ("injection.role_override", re.compile(r"you\s+are\s+now\b", re.IGNORECASE)),
    ("injection.role_override", re.compile(r"pretend\s+(?:to\s+be|you\s+are)\b", re.IGNORECASE)),
    ("injection.instruction_override", re.compile(r"(?:以前|前|上記|これまで)の(?:指示|命令|ルール)を(?:無視|忘れ)")),
    ("injection.prompt_disclosure", re.compile(r"システムプロンプト")),
    ("injection.instruction_override", re.compile(r"指示を無視")),
]

# Chat-template control tokens, screened as a CLASS rather than as a list of
# known model dialects. An attack carried in this form contains no directive
# phrase at all, so a phrase-only screen passes it straight through.
_CONTROL_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("injection.control_token", re.compile(r"<\|[^|>]{0,64}\|>")),
    ("injection.control_token", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("injection.control_token", re.compile(r"<</?SYS>>", re.IGNORECASE)),
    ("injection.control_token", re.compile(r"<\|?(?:im_start|im_end|endoftext|system)\|?>", re.IGNORECASE)),
]

Finding = tuple[str, str]  # (category, reason_code)


def normalise_for_screening(text: str) -> str:
    """Return the compatibility-normalised form of *text*.

    Full-width punctuation and digits are the ordinary way to type Japanese
    clinical text, and they are what the half-width patterns miss.
    """
    return unicodedata.normalize("NFKC", text)


def _scan_patterns(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> Optional[str]:
    for code, pattern in patterns:
        if pattern.search(text):
            return code
    return None


def _forms(text: str) -> tuple[str, ...]:
    """The original text plus its normalised form, de-duplicated."""
    normalised = normalise_for_screening(text)
    return (text,) if normalised == text else (text, normalised)


def scan_for_protected_identifiers(text: str) -> Optional[str]:
    """Reason code for the first protected-identifier match, else None."""
    for form in _forms(text):
        code = _scan_patterns(form, _PROTECTED_ID_PATTERNS)
        if code is not None:
            return code
        for finding in detect_pii(form):
            if finding["type"] not in _HEURISTIC_PII_TYPES:
                return f"protected_id.{finding['type']}"
    return None


def scan_for_injection(text: str) -> Optional[str]:
    """Reason code for the first injection match, else None."""
    for form in _forms(text):
        code = _scan_patterns(form, _CONTROL_TOKEN_PATTERNS)
        if code is not None:
            return code
        code = _scan_patterns(form, _INJECTION_PHRASE_PATTERNS)
        if code is not None:
            return code
    return None


def contains_credentials(value: Any) -> bool:
    """True when *value* carries a credential shape the platform would block.

    Delegated to the framework detector rather than kept as a local pattern
    list: a local set narrower than the platform's is a bypass, because the
    value passes this screen, the platform gate then raises inside the node
    wrapper, and the wrapper discards the node's whole delta — including any
    clearing this template performed.
    """
    return bool(detect_credentials_in_value(value))


def scan_for_threats(text: str) -> Optional[Finding]:
    """Screen *text* for injection and protected identifiers.

    Returns ``(category, reason_code)`` on the first finding — ``category`` is
    ``"injection"``, ``"protected_id"`` or ``"credential"`` — or None when the
    text is clean. The reason code is a stable identifier; the matched span is
    never returned, logged or echoed.
    """
    if not isinstance(text, str) or not text:
        return None
    code = scan_for_injection(text)
    if code is not None:
        return ("injection", code)
    code = scan_for_protected_identifiers(text)
    if code is not None:
        return ("protected_id", code)
    if contains_credentials(text):
        return ("credential", "credential.detected")
    return None
