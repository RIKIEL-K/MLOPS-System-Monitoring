"""
tests/test_ingest.py — Tests unitaires pour steps/ingest.py (classe Ingestion)
"""

import os
import pandas as pd
import pytest

from steps.ingest import Ingestion, REQUIRED_COLUMNS

def make_valid_csv(tmp_path, n_rows: int = 50) -> str:
    df = pd.DataFrame({
        "message":          [f"GET /endpoint/{i} status=200" for i in range(n_rows)],
        "level":            ["INFO"] * n_rows,
        "status_code":      [200]  * n_rows,
        "method":           ["GET"] * n_rows,
        "endpoint":         [f"/api/{i}" for i in range(n_rows)],
        "component":        ["api"]     * n_rows,
        "action":           ["request"] * n_rows,
        "response_time_ms": [10.0]  * n_rows,
    })
    path = str(tmp_path / "train.csv")
    df.to_csv(path, index=False)
    return path


def make_config(tmp_path, train_path: str, test_path: str | None = None) -> str:
    """Générer un config.yml minimal pour les tests."""
    import yaml
    test_path = test_path or str(tmp_path / "test.csv")
    cfg = {
        "paths": {"train_data": train_path, "test_data": test_path, "models_dir": str(tmp_path / "models")},
        "mlflow": {"tracking_uri": str(tmp_path / "mlruns"), "experiment_name": "test", "model_name": "test", "artifact_path": "model"},
        "tfidf":  {"max_features": 50, "min_df": 1, "max_df": 0.95},
        "kmeans": {"n_clusters": 2, "n_init": 3, "random_state": 42, "k_range": "2,3"},
        "split":  {"test_size": 0.2, "random_state": 42},
        "api":    {"host": "0.0.0.0", "port": 8000},
        "monitoring": {"reference_data": "", "production_data": ""},
    }
    config_path = str(tmp_path / "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return config_path


class TestIngestion:

    def test_load_data_returns_two_dataframes(self, tmp_path):
        train_path = make_valid_csv(tmp_path, 50)
        test_path  = make_valid_csv(tmp_path, 20)
        # renommer pour avoir train et test distincts
        import shutil
        shutil.copy(train_path, str(tmp_path / "train.csv"))
        shutil.copy(test_path,  str(tmp_path / "test.csv"))
        config_path = make_config(tmp_path, str(tmp_path / "train.csv"), str(tmp_path / "test.csv"))
        ing = Ingestion(config_path=config_path)
        train, test = ing.load_data()
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test,  pd.DataFrame)

    def test_required_columns_present(self, tmp_path):
        path = make_valid_csv(tmp_path)
        import shutil
        shutil.copy(path, str(tmp_path / "train.csv"))
        shutil.copy(path, str(tmp_path / "test.csv"))
        config_path = make_config(tmp_path, str(tmp_path / "train.csv"), str(tmp_path / "test.csv"))
        ing = Ingestion(config_path=config_path)
        train, _ = ing.load_data()
        for col in REQUIRED_COLUMNS:
            assert col in train.columns

    def test_missing_file_raises(self, tmp_path):
        config_path = make_config(tmp_path, "nonexistent/train.csv", "nonexistent/test.csv")
        ing = Ingestion(config_path=config_path)
        with pytest.raises(FileNotFoundError):
            ing.load_data()

    def test_missing_column_raises(self, tmp_path):
        bad_df = pd.DataFrame({"level": ["INFO"], "status_code": [200]})
        bad_path = str(tmp_path / "train.csv")
        bad_df.to_csv(bad_path, index=False)
        shutil2_path = str(tmp_path / "test.csv")
        bad_df.to_csv(shutil2_path, index=False)
        config_path = make_config(tmp_path, bad_path, shutil2_path)
        ing = Ingestion(config_path=config_path)
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            ing.load_data()

    def test_status_code_coerced_to_int(self, tmp_path):
        path = make_valid_csv(tmp_path)
        import shutil
        shutil.copy(path, str(tmp_path / "train.csv"))
        shutil.copy(path, str(tmp_path / "test.csv"))
        config_path = make_config(tmp_path, str(tmp_path / "train.csv"), str(tmp_path / "test.csv"))
        ing = Ingestion(config_path=config_path)
        train, _ = ing.load_data()
        assert train["status_code"].dtype in [int, "int64", "int32"]
