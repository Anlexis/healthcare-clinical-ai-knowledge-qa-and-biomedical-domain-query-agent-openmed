"""Unit tests for the runtime-settings resolver.

Every caller- or operator-controlled number in this template goes through one
parser, so the non-finite matrix is exercised once, per field.
"""

import pytest

from src.services.runtime_settings import (
    RUNTIME_SETTINGS_KEY,
    SettingsError,
    resolve_runtime_settings,
    settings_from_state,
)

# (config path, a value inside the permitted range)
FIELDS = [
    (("query", "max_chars"), "query_max_chars", 500),
    (("retrieval", "top_k"), "retrieval_top_k", 5),
    (("retrieval", "query_excerpt_chars"), "retrieval_query_excerpt_chars", 40),
    (("retrieval", "answer_excerpt_chars"), "answer_query_excerpt_chars", 60),
    (("clinical", "max_guidelines"), "max_guidelines", 1),
    (("clinical", "max_icd11_codes"), "max_icd11_codes", 2),
    (("clinical", "max_context_terms"), "max_context_terms", 4),
]

NON_FINITE_VALUES = [
    "NaN",
    "nan",
    "Infinity",
    "-Infinity",
    float("nan"),
    float("inf"),
    float("-inf"),
]

NON_NUMERIC_VALUES = ["", "abc", None, [], {}, True, False]


def _nested(path, value):
    return {path[0]: {path[1]: value}}


class TestDefaults:
    def test_empty_config_yields_documented_defaults(self):
        resolved = resolve_runtime_settings({})
        assert resolved["retrieval_top_k"] == 3
        assert resolved["query_max_chars"] == 2000
        assert resolved["max_context_terms"] == 8

    def test_none_config_is_accepted(self):
        assert resolve_runtime_settings(None)["retrieval_top_k"] == 3


class TestDeclaredValuesAreUsed:
    @pytest.mark.parametrize("path,field,value", FIELDS)
    def test_declared_value_reaches_the_resolver(self, path, field, value):
        assert resolve_runtime_settings(_nested(path, value))[field] == value

    @pytest.mark.parametrize("path,field,value", FIELDS)
    def test_string_form_is_accepted(self, path, field, value):
        assert resolve_runtime_settings(_nested(path, str(value)))[field] == value


class TestFailClosed:
    @pytest.mark.parametrize("path,field,_value", FIELDS)
    @pytest.mark.parametrize("bad", NON_FINITE_VALUES)
    def test_non_finite_values_are_refused(self, path, field, _value, bad):
        """NaN and the infinities parse fine and then compare False against every
        bound, so an unchecked value silently stops bounding anything."""
        with pytest.raises(SettingsError) as excinfo:
            resolve_runtime_settings(_nested(path, bad))
        assert ".".join(path) in str(excinfo.value)

    @pytest.mark.parametrize("path,field,_value", FIELDS)
    @pytest.mark.parametrize("bad", NON_NUMERIC_VALUES)
    def test_non_numeric_values_are_refused(self, path, field, _value, bad):
        with pytest.raises(SettingsError):
            resolve_runtime_settings(_nested(path, bad))

    @pytest.mark.parametrize("path,field,_value", FIELDS)
    def test_out_of_range_magnitudes_are_refused(self, path, field, _value):
        with pytest.raises(SettingsError):
            resolve_runtime_settings(_nested(path, 10**9))

    def test_fractional_values_are_refused(self):
        with pytest.raises(SettingsError):
            resolve_runtime_settings({"retrieval": {"top_k": 2.5}})

    def test_a_block_declared_as_a_scalar_is_refused(self):
        with pytest.raises(SettingsError):
            resolve_runtime_settings({"retrieval": 3})

    def test_error_message_names_the_field_not_the_value(self):
        with pytest.raises(SettingsError) as excinfo:
            resolve_runtime_settings({"retrieval": {"top_k": "9999999"}})
        assert "retrieval.top_k" in str(excinfo.value)
        assert "9999999" not in str(excinfo.value)


class TestStateSeeding:
    def test_settings_are_read_back_from_state(self):
        seeded = resolve_runtime_settings({"retrieval": {"top_k": 7}})
        assert settings_from_state({RUNTIME_SETTINGS_KEY: seeded})["retrieval_top_k"] == 7

    def test_absent_seed_falls_back_to_defaults(self):
        assert settings_from_state({})["retrieval_top_k"] == 3
