"""
app.py — API FastAPI pour le serving du modèle de clustering de logs

Expose deux endpoints :
  - GET  /health  → Santé de l'API
  - POST /predict → Prédiction de cluster(s) pour un ou plusieurs messages

Usage:
    uvicorn app:app --reload
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="AIOps Log Clustering API",
    description=(
        "API de clustering de logs via TF-IDF + K-Means. "
        "Envoie des messages de logs pour obtenir leur cluster et label."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS permissif pour le développement local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas Pydantic ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    messages: List[str] = Field(
        ...,
        min_items=1,
        description="Liste de messages de logs à classifier",
        example=[
            "level=error msg='connection refused' service=api-gateway",
            "GET /health HTTP/1.1 status=200 response_time=5ms",
        ],
    )


class PredictionItem(BaseModel):
    message:       str
    cluster_id:    int
    cluster_label: str


class PredictResponse(BaseModel):
    predictions: List[PredictionItem]
    model:       str = "log-clustering-kmeans"
    n_clusters:  int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health_check() -> dict:
    """Vérifier que l'API est opérationnelle."""
    return {"status": "ok", "service": "AIOps Log Clustering API"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_logs(request: PredictRequest) -> PredictResponse:
    """
    Classifier un ou plusieurs messages de logs.

    Retourne pour chaque message son cluster_id et un label interprétatif.
    """
    try:
        from steps.predict import predict
        results_df = predict(request.messages)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    predictions = [
        PredictionItem(
            message=msg,
            cluster_id=int(row["cluster_id"]),
            cluster_label=str(row["cluster_label"]),
        )
        for msg, (_, row) in zip(request.messages, results_df.iterrows())
    ]

    return PredictResponse(
        predictions=predictions,
        n_clusters=results_df["cluster_id"].nunique(),
    )


@app.get("/", tags=["Info"])
def root() -> dict:
    return {
        "message": "AIOps Log Clustering API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict",
    }
