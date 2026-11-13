"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import DEFAULT_DATA_FILE, DEFAULT_MODEL_FILE
from app.model import IrisModel
from app.repository import PredictionRepository
from app.routers import predictions
from app.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TAGS_METADATA = [
    {
        "name": "predictions",
        "description": (
            "Ejecute el clasificador Iris y administre los registros de "
            "predicciones almacenados."
        ),
    },
    {
        "name": "health",
        "description": "Comprobaciones de vida del servicio.",
    },
]

DESCRIPTION = """
Servicio FastAPI mínimo que clasifica flores Iris con un modelo de
scikit-learn y expone un CRUD pequeño sobre las predicciones almacenadas.

**Flujo**

1. `POST /api/v1/predict` ejecuta el modelo y persiste el resultado.
2. `GET /api/v1/predictions` lista el historial.
3. `PUT /api/v1/predictions/{id}` vuelve a ejecutar el modelo con nuevas medidas.
4. `DELETE /api/v1/predictions/{id}` elimina un registro.

Los datos se almacenan en un archivo JSON local, por lo que no hay una base
de datos externa que ejecutar. La documentación interactiva está disponible
en `/docs` (Swagger UI) y `/redoc`.
"""


def create_app(
    data_file: Path | str | None = None,
    model_file: Path | str | None = None,
) -> FastAPI:
    """Build the application.

    Paths are injectable so tests can use temporary files instead of the real
    `data/` and `models/` directories.
    """
    data_path = Path(data_file) if data_file is not None else DEFAULT_DATA_FILE
    model_path = Path(model_file) if model_file is not None else DEFAULT_MODEL_FILE

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = IrisModel(model_path)
        model.load_or_train()

        app.state.model = model
        app.state.repository = PredictionRepository(data_path)

        logger.info("API ready. Swagger UI: http://127.0.0.1:8000/docs")
        yield

    app = FastAPI(
        title="API de Predicción de Iris",
        description=DESCRIPTION,
        version="1.0.0",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    app.include_router(predictions.router)

    @app.get(
        "/health",
        tags=["health"],
        summary="Comprobación de salud",
        response_model=HealthResponse,
    )
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
