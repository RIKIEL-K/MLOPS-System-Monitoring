
PYTHON  := python
PIP     := pip
UVICORN := uvicorn
CONFIG  := config.yml

.PHONY: setup dataset train serve mlflow-ui test docs \
        dvc-init dvc-repro dvc-exp dvc-metrics dvc-plots lint clean help

help:
	@echo ""
	@echo "  AIOps Log Clustering — Commandes disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup        Installer les dépendances Python"
	@echo "  make dataset      Générer data/train.csv et data/test.csv"
	@echo "  make train        Entraîner le modèle (MLflow local)"
	@echo "  make serve        Lancer l'API FastAPI (port 8000)"
	@echo "  make mlflow-ui    Ouvrir l'interface MLflow (port 5000)"
	@echo "  make test         Lancer les tests pytest"
	@echo "  make docs         Lancer la documentation MkDocs (port 8001)"
	@echo ""
	@echo "  ── DVC ───────────────────────────────────────────"
	@echo "  make dvc-init     Initialiser DVC + tracker les données"
	@echo "  make dvc-repro    Relancer le pipeline (si changements)"
	@echo "  make dvc-exp      Lancer une expérience DVC (ex: n_clusters=8)"
	@echo "  make dvc-metrics  Voir les métriques de toutes les expériences"
	@echo "  make dvc-plots    Visualiser la distribution des clusters"
	@echo "  ──────────────────────────────────────────────────"
	@echo "  make lint         Vérifier le style du code"
	@echo "  make clean        Nettoyer les fichiers temporaires"
	@echo ""

setup:
	$(PIP) install -r requirements.txt

dataset:
	$(PYTHON) dataset.py

train:
	$(PYTHON) main.py --config $(CONFIG)

serve:
	$(UVICORN) app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest tests/ -v --tb=short

docs:
	mkdocs serve --dev-addr 0.0.0.0:8001

dvc-init:
	dvc init
	dvc add data/mock_loki_logs.csv
	@echo "DVC initialisé et données trackées."
	@echo "Configurez un remote : dvc remote add -d myremote <path>"

dvc-repro:
	dvc repro

dvc-exp:
	@echo "Exemple : dvc exp run -S kmeans.n_clusters=8"
	@echo "Exemple : dvc exp run -S tfidf.max_features=150 -S kmeans.n_clusters=8"
	dvc exp run

dvc-metrics:
	dvc metrics show
	dvc metrics diff

dvc-plots:
	dvc plots show plots/cluster_distribution.csv

