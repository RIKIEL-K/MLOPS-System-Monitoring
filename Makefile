
PYTHON  := python
PIP     := pip
UVICORN := uvicorn
CONFIG  := config.yml

.PHONY: setup dataset train serve mlflow-ui test docs \
        dvc-init dvc-push lint clean help

help:
	@echo ""
	@echo "  AIOps Log Clustering — Commandes disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup       Installer les dépendances Python"
	@echo "  make dataset     Générer data/train.csv et data/test.csv"
	@echo "  make train       Entraîner le modèle (MLflow local)"
	@echo "  make serve       Lancer l'API FastAPI (port 8000)"
	@echo "  make mlflow-ui   Ouvrir l'interface MLflow (port 5000)"
	@echo "  make test        Lancer les tests pytest"
	@echo "  make docs        Lancer la documentation MkDocs (port 8001)"
	@echo "  make dvc-init    Initialiser DVC dans le projet"
	@echo "  make dvc-push    Versionner les données (DVC)"
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
	@echo "DVC initialisé. Configurez un remote avec : dvc remote add -d myremote <path>"

dvc-push:
	dvc add data/train.csv data/test.csv
	dvc push
	@echo "Données versionnées et poussées vers le remote DVC."
