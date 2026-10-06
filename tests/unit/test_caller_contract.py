"""Unit tests for the structured caller-context contract."""

import pytest

from framework.security.credential_detector import detect_credentials_in_value
from src.services.caller_contract import (
    CONTEXT_FIELDS,
    drop_unknown_context_fields,
    validate_caller_context,
)


def _validate(raw, max_terms=8):
    return validate_caller_context(raw, max_terms=max_terms)


class TestAcceptedShapes:
    def test_absent_context_is_valid(self):
        assert _validate(None) == ({}, None)

    def test_empty_context_is_valid(self):
        assert _validate({}) == ({}, None)

    def test_declared_fields_are_kept(self):
        validated, error = _validate({"channel": "portal", "clinical_terms": ["warfarin", "aspirin"]})
        assert error is None
        assert validated == {"channel": "portal", "clinical_terms": ["warfarin", "aspirin"]}

    def test_inert_alphabet_is_accepted(self):
        validated, error = _validate({"clinical_terms": ["diabetes_mellitus_type_2", "icd-11", "a1c"]})
        assert error is None
        assert len(validated["clinical_terms"]) == 3


class TestRejectedShapes:
    def test_non_mapping_is_refused(self):
        assert _validate("a string")[1] == "input_context: must be an object"

    def test_unknown_field_is_refused(self):
        validated, error = _validate({"surprise": "value"})
        assert validated == {}
        assert error == "input_context.surprise: unrecognised field"

    def test_unsafe_field_name_is_reported_by_position_only(self):
        validated, error = _validate({"AKIA" + "IOSFODNN7EXAMPLE!!": "x"})
        assert validated == {}
        assert error == "input_context field #1: unrecognised field"
        assert "AKIA" not in error

    def test_free_text_term_is_refused(self):
        assert "inert identifier" in _validate({"clinical_terms": ["Type 2 Diabetes Mellitus"]})[1]

    def test_over_length_term_is_refused(self):
        assert _validate({"clinical_terms": ["a" * 33]})[1] is not None

    def test_term_cap_is_enforced(self):
        error = _validate({"clinical_terms": ["a", "b", "c"]}, max_terms=2)[1]
        assert "at most 2 entries" in error

    def test_wrong_types_are_refused(self):
        assert _validate({"channel": 7})[1].endswith("must be a string")
        assert _validate({"clinical_terms": "warfarin"})[1].endswith("must be a list")
        assert _validate({"clinical_terms": [7]})[1].endswith("must be a string")


class TestScreensOnTheContextChannel:
    """The platform input filter does not scan this channel, so the template
    must — for every class the free-text question is screened for."""

    def test_credential_shape_is_refused(self):
        validated, error = _validate({"channel": "AKIA" + "IOSFODNN7EXAMPLE"})
        assert validated == {}
        assert error == "input_context.channel: credential-shaped value refused"

    def test_protected_identifier_is_refused(self):
        error = _validate({"clinical_terms": ["1234567890"]})[1]
        assert "protected identifier refused" in error

    def test_injection_pattern_is_refused(self):
        error = _validate({"channel": "<|im_start|>"})[1]
        assert "instruction-injection pattern refused" in error

    def test_rejection_never_echoes_the_value(self):
        secret = "sk_live_" + "abcdefghijklmnopqrstuvwx"
        error = _validate({"channel": secret})[1]
        assert secret not in error

    def test_refusal_set_matches_the_platform_detector(self):
        """Property: a value is refused for the credential reason exactly when
        the platform detector finds a credential in it. Pinning the identity
        prevents this template's set from drifting narrower than the gate that
        runs immediately after it."""
        candidates = [
            "AKIA" + "IOSFODNN7EXAMPLE",
            "sk_live_" + "abcdefghijklmnopqrstuvwx",
            "warfarin",
            "icd-11",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ]
        for value in candidates:
            error = _validate({"channel": value})[1]
            refused_as_credential = error is not None and "credential-shaped" in error
            assert refused_as_credential == bool(detect_credentials_in_value(value))


class TestDropUnknownFields:
    def test_unknown_keys_are_dropped_not_carried(self):
        """Dropping is what makes the channel safe by construction: a key that
        is merely ignored still reaches graph state."""
        cleaned = drop_unknown_context_fields({"channel": "portal", "surprise": "AKIA" + "IOSFODNN7EXAMPLE"})
        assert cleaned == {"channel": "portal"}

    def test_non_mapping_becomes_empty(self):
        assert drop_unknown_context_fields("nope") == {}

    def test_every_declared_field_survives(self):
        raw = {name: "x" for name in CONTEXT_FIELDS}
        assert set(drop_unknown_context_fields(raw)) == set(CONTEXT_FIELDS)


@pytest.mark.parametrize("field", CONTEXT_FIELDS)
def test_declared_fields_are_documented(field):
    assert field in {"channel", "clinical_terms"}
