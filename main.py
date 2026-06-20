"""
main.py — Orchestrateur du pipeline ML (local)

Exécute le pipeline dans l'ordre :
  1. Ingest   → charger train.csv et test.csv
  2. Clean    → nettoyer et extraire les log_patterns
  3. Prepare  → valider et afficher les stats du dataset
  4. Train    → TF-IDF + K-Means + MLflow local + sauvegarde
  5. Evaluate → silhouette, anomaly_rate sur le jeu de test
  6. Results  → afficher + exporter metrics.json et plots/ (DVC)

Usage:
    python main.py                  ← exécution directe
    dvc repro                       ← via pipeline DVC
    dvc exp run -S kmeans.n_clusters=8  ← expérience DVC
"""

import logging
import argparse

from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline AIOps Log Clustering — local + DVC",
    )
    parser.add_argument(
        "--config", type=str, default="config.yml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--params", type=str, default="params.yaml",
        help="Chemin vers le fichier de paramètres DVC",
    )
    args = parser.parse_args()

    # ── 1. Ingest ──────────────────────────────────────────────────────
    ingestion = Ingestion(config_path=args.config, params_path=args.params)
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    # ── 2. Clean ───────────────────────────────────────────────────────
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data  = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    # ── 3. Prepare ─────────────────────────────────────────────────────
    n_patterns = train_data["log_pattern"].nunique()
    n_train    = len(train_data)
    n_test     = len(test_data)
    logging.info(
        "Data preparation completed — train: %d rows | test: %d rows | patterns: %d unique",
        n_train, n_test, n_patterns,
    )

    # ── 4. Train ───────────────────────────────────────────────────────
    trainer = Trainer(config_path=args.config, params_path=args.params)
    trainer.train(train_data)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # ── 5. Evaluate ────────────────────────────────────────────────────
    predictor = Predictor(config_path=args.config)
    results = predictor.evaluate(test_data)
    logging.info("Model evaluation completed successfully")

    # ── 6. Results — affichage + export DVC ────────────────────────────
    trainer.save_metrics(extra={
        "test_silhouette_score": results["silhouette_score"],
        "test_anomaly_rate":     results["anomaly_rate"],
        "n_samples_evaluated":   results["n_samples_evaluated"],
    })
    trainer.save_plots(results["cluster_distribution"])

    print("\n============= Model Evaluation Results ==============")
    print(f"Model            : {trainer.model_name}")
    print(f"Silhouette Score : {results['silhouette_score']:.4f}")
    print(f"Anomaly Rate     : {results['anomaly_rate']:.2%}")
    print(f"Clusters         : {results['n_clusters']}")
    print(f"Samples évalués  : {results['n_samples_evaluated']:,}")
    print("\nDistribution des clusters :")
    for label, count in results["cluster_distribution"].items():
        pct = count / results["n_samples_evaluated"] * 100
        print(f"  {label:<45} {count:>5} logs  ({pct:.1f}%)")
    print("=====================================================\n")
    print("DVC  : dvc metrics show  →  voir metrics.json")
    print("DVC  : dvc plots show    →  voir cluster_distribution.csv")
    print("DVC  : dvc exp run -S kmeans.n_clusters=8  →  nouvelle expérience")
    print(f"MLflow : mlflow ui  →  http://localhost:5000")
    print(f"API    : uvicorn app:app --reload  →  http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
