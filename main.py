"""
main.py — Orchestrateur du pipeline ML (local)

Remplace le pipeline Kubeflow. Exécute les étapes dans l'ordre :
  1. (Optionnel) Générer le dataset — python dataset.py
  2. Entraîner le modèle — steps/train.py

Usage:
    python main.py
    python main.py --config config.yml
    python main.py --skip-dataset
"""

import argparse
import os
import sys
import time
import yaml


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        print(f"[ERROR] config.yml introuvable : {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_data_exists(config: dict) -> bool:
    """Vérifier si train.csv existe déjà."""
    train_path = config["paths"]["train_data"]
    return os.path.isfile(train_path)


def run_pipeline(config: dict, skip_dataset: bool = False) -> None:
    """
    Exécuter le pipeline complet : dataset → train.

    Args:
        config:       Configuration chargée depuis config.yml.
        skip_dataset: Si True, sauter la génération du dataset.
    """
    print("=" * 60)
    print("  AIOps Log Clustering — Pipeline Local")
    print("=" * 60)
    start_time = time.time()

    # ── Étape 0 : Générer le dataset si nécessaire ─────────────────────
    if not skip_dataset and not check_data_exists(config):
        print("\n[Step 0] Génération du dataset (train.csv + test.csv)")
        import subprocess
        result = subprocess.run(
            [sys.executable, "dataset.py"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("[ERROR] Échec de la génération du dataset.")
            sys.exit(1)
    elif check_data_exists(config):
        print(f"\n[Step 0] Dataset existant détecté — {config['paths']['train_data']}")
    else:
        print("\n[Step 0] Génération du dataset ignorée (--skip-dataset)")

    # ── Étape 1 : Entraînement ─────────────────────────────────────────
    print("\n[Step 1] Lancement de l'entraînement TF-IDF + K-Means")
    from steps.train import train_model
    run_id = train_model(config)

    # ── Résumé ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"{'=' * 60}")
    print(f"  MLflow run ID : {run_id}")
    print(f"  Durée totale  : {elapsed:.1f}s")
    print(f"  Interface     : mlflow ui  (http://localhost:5000)")
    print(f"  API serving   : uvicorn app:app --reload")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestre le pipeline ML de clustering de logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        default=False,
        help="Ne pas régénérer le dataset si train.csv existe déjà",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, skip_dataset=args.skip_dataset)


if __name__ == "__main__":
    main()
