"""
tests/test_ingest.py — Tests unitaires pour steps/ingest.py
"""

import os
import tempfile

import pandas as pd
import pytest

from steps.ingest import load_data, get_data_stats, REQUIRED_COLUMNS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_valid_csv(tmp_path, n_rows: int = 50) -> str:
    """Créer un CSV valide minimal pour les tests."""
    df = pd.DataFrame({
        "message":         [f"GET /endpoint/{i} status=200" for i in range(n_rows)],
        "level":           ["INFO"] * n_rows,
        "status_code":     [200] * n_rows,
        "method":          ["GET"] * n_rows,
        "endpoint":        [f"/api/{i}" for i in range(n_rows)],
        "component":       ["api"] * n_rows,
        "action":          ["request"] * n_rows,
        "response_time_ms":[10.0] * n_rows,
    })
    path = str(tmp_path / "test_data.csv")
    df.to_csv(path, index=False)
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadData:

    def test_load_valid_csv(self, tmp_path):
        path = make_valid_csv(tmp_path)
        df = load_data(path, validate=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_required_columns_present(self, tmp_path):
        path = make_valid_csv(tmp_path)
        df = load_data(path)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent/path/file.csv")

    def test_missing_column_raises(self, tmp_path):
        """Un CSV sans la colonne 'message' doit lever une ValueError."""
        df = pd.DataFrame({"level": ["INFO"], "status_code": [200]})
        path = str(tmp_path / "bad.csv")
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            load_data(path, validate=True)

    def test_drops_null_messages(self, tmp_path):
        """Les lignes avec message=NaN doivent être supprimées."""
        df = pd.DataFrame({
            "message":         [None, "ok message", None],
            "level":           ["INFO"] * 3,
            "status_code":     [200] * 3,
            "method":          ["GET"] * 3,
            "endpoint":        ["/"] * 3,
            "component":       ["api"] * 3,
            "action":          ["req"] * 3,
            "response_time_ms":[5.0] * 3,
        })
        path = str(tmp_path / "nulls.csv")
        df.to_csv(path, index=False)
        result = load_data(path, validate=True)
        assert len(result) == 1

    def test_status_code_coerced_to_int(self, tmp_path):
        path = make_valid_csv(tmp_path)
        df = load_data(path)
        assert df["status_code"].dtype in [int, "int64", "int32"]


class TestGetDataStats:

    def test_returns_expected_keys(self, tmp_path):
        path = make_valid_csv(tmp_path)
        df = load_data(path)
        stats = get_data_stats(df)
        assert "n_rows" in stats
        assert "level_distribution" in stats
        assert "avg_response_time_ms" in stats

    def test_n_rows_correct(self, tmp_path):
        path = make_valid_csv(tmp_path, n_rows=30)
        df = load_data(path)
        stats = get_data_stats(df)
        assert stats["n_rows"] == 30
