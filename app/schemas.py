"""Pydantic models that define the API contract and Swagger examples."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

IrisLabel = Literal["setosa", "versicolor", "virginica"]

_FEATURE_EXAMPLE = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


class PredictionFeatures(BaseModel):
    """Medidas utilizadas por el modelo (en centímetros)."""

    model_config = ConfigDict(
        json_schema_extra={"example": _FEATURE_EXAMPLE},
    )

    sepal_length: float = Field(
        ...,
        gt=0,
        le=30,
        description="Longitud del sépalo en centímetros.",
        examples=[5.1],
    )
    sepal_width: float = Field(
        ...,
        gt=0,
        le=30,
        description="Ancho del sépalo en centímetros.",
        examples=[3.5],
    )
    petal_length: float = Field(
        ...,
        gt=0,
        le=30,
        description="Longitud del pétalo en centímetros.",
        examples=[1.4],
    )
    petal_width: float = Field(
        ...,
        gt=0,
        le=30,
        description="Ancho del pétalo en centímetros.",
        examples=[0.2],
    )


class PredictionCreate(PredictionFeatures):
    """Cuerpo de la solicitud para una nueva predicción."""

    model_config = ConfigDict(
        json_schema_extra={"example": _FEATURE_EXAMPLE},
    )


class PredictionRecord(PredictionFeatures):
    """Predicción almacenada que devuelve la API."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "predicted_class": 0,
                "predicted_label": "setosa",
                "confidence": 0.981,
                "created_at": "2026-01-01T12:00:00+00:00",
                "updated_at": "2026-01-01T12:00:00+00:00",
            }
        }
    )

    id: int = Field(..., description="Identificador de la predicción.", examples=[1])
    predicted_class: int = Field(
        ...,
        ge=0,
        le=2,
        description="Clase numérica predicha por el modelo.",
        examples=[0],
    )
    predicted_label: IrisLabel = Field(
        ...,
        description="Nombre legible de la especie.",
        examples=["setosa"],
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilidad de la clase predicha.",
        examples=[0.981],
    )
    created_at: datetime = Field(..., description="Fecha y hora de creación (UTC).")
    updated_at: datetime = Field(
        ...,
        description="Fecha y hora de la última actualización (UTC).",
    )


class HealthResponse(BaseModel):
    """Respuesta de la comprobación de salud."""

    status: str = Field(..., description="Estado del servicio.", examples=["ok"])
