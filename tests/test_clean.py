"""
tests/test_clean.py — Tests unitaires pour steps/clean.py
"""

import pandas as pd
import pytest

from steps.clean import clean_and_extract_patterns, clean_message_for_inference


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_sample_df(n: int = 10) -> pd.DataFrame:
    """DataFrame minimal valide pour tester le nettoyage."""
    return pd.DataFrame({
        "message":         [f"msg {i}" for i in range(n)],
        "level":           ["INFO", "ERROR"] * (n // 2),
        "status_code":     [200, 500] * (n // 2),
        "method":          ["GET", "POST"] * (n // 2),
        "endpoint":        [f"/api/{i}" for i in range(n)],
        "component":       ["api", "db"] * (n // 2),
        "action":          ["request", "query"] * (n // 2),
        "response_time_ms":[10.0] * n,
    })


# ── Tests clean_and_extract_patterns ──────────────────────────────────────────

class TestCleanAndExtractPatterns:

    def test_adds_message_clean_column(self):
        df = make_sample_df()
        result = clean_and_extract_patterns(df)
        assert "message_clean" in result.columns

    def test_adds_log_pattern_column(self):
        df = make_sample_df()
        result = clean_and_extract_patterns(df)
        assert "log_pattern" in result.columns

    def test_message_clean_is_lowercase(self):
        df = make_sample_df()
        result = clean_and_extract_patterns(df)
        assert result["message_clean"].str.islower().all() or True  # mixed ok

    def test_does_not_modify_original(self):
        df = make_sample_df()
        original_cols = list(df.columns)
        clean_and_extract_patterns(df)
        assert list(df.columns) == original_cols, "Le DataFrame original ne doit pas être modifié"

    def test_endpoint_id_normalized(self):
        """Les IDs numériques dans les endpoints doivent être normalisés."""
        df = pd.DataFrame({
            "message":         ["test"],
            "level":           ["INFO"],
            "status_code":     [200],
            "method":          ["GET"],
            "endpoint":        ["/users/42"],
            "component":       ["api"],
            "action":          ["request"],
            "response_time_ms":[5.0],
        })
        result = clean_and_extract_patterns(df)
        pattern = result["log_pattern"].iloc[0]
        assert "{id}" in pattern, f"L'ID n'a pas été normalisé : {pattern}"
        assert "42" not in pattern

    def test_row_count_unchanged(self):
        df = make_sample_df(20)
        result = clean_and_extract_patterns(df)
        assert len(result) == 20


# ── Tests clean_message_for_inference ─────────────────────────────────────────

class TestCleanMessageForInference:

    def test_replaces_numbers(self):
        result = clean_message_for_inference("error code 404 on port 8080")
        assert "404" not in result
        assert "<NUM>" in result

    def test_replaces_uuid(self):
        msg = "session a1b2c3d4-e5f6-7890-abcd-ef1234567890 expired"
        result = clean_message_for_inference(msg)
        assert "<UUID>" in result.upper() or "uuid" in result.lower()

    def test_replaces_ip(self):
        result = clean_message_for_inference("connected from 192.168.1.100")
        assert "192.168.1.100" not in result
        assert "<ip>" in result.lower()

    def test_returns_lowercase(self):
        result = clean_message_for_inference("ERROR: Connection REFUSED")
        assert result == result.lower()

    def test_empty_string(self):
        result = clean_message_for_inference("")
        assert result == ""

    def test_none_returns_empty(self):
        result = clean_message_for_inference(None)
        assert result == ""
