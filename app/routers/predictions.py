"""Prediction and CRUD endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_model, get_repository
from app.model import IrisModel
from app.repository import PredictionNotFoundError, PredictionRepository
from app.schemas import PredictionCreate, PredictionRecord

router = APIRouter(prefix="/api/v1", tags=["predictions"])

_NOT_FOUND_RESPONSE = {
    404: {
        "description": "Predicción no encontrada.",
        "content": {
            "application/json": {
                "example": {"detail": "Predicción 999 no encontrada"},
            }
        },
    }
}


def _features_list(payload: PredictionCreate) -> list[float]:
    return [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]


@router.post(
    "/predict",
    response_model=PredictionRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Predecir la especie de Iris y guardar el resultado",
    description=(
        "Ejecuta el modelo entrenado sobre las cuatro medidas, persiste la "
        "predicción y devuelve el registro almacenado."
    ),
    responses={
        422: {
            "description": "Error de validación (valores faltantes o fuera de rango).",
        }
    },
)
def predict(
    payload: PredictionCreate,
    model: Annotated[IrisModel, Depends(get_model)],
    repository: Annotated[PredictionRepository, Depends(get_repository)],
) -> PredictionRecord:
    predicted_class, predicted_label, confidence = model.predict(_features_list(payload))
    return repository.create(
        features=payload.model_dump(),
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        confidence=confidence,
    )


@router.get(
    "/predictions",
    response_model=list[PredictionRecord],
    summary="Listar las predicciones almacenadas",
    description="Devuelve las predicciones almacenadas, las más recientes primero.",
)
def list_predictions(
    repository: Annotated[PredictionRepository, Depends(get_repository)],
    skip: Annotated[int, Query(ge=0, description="Registros a omitir.")] = 0,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Cantidad máxima de registros a devolver."),
    ] = 50,
) -> list[PredictionRecord]:
    return repository.list_predictions(skip=skip, limit=limit)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecord,
    summary="Obtener una predicción",
    responses=_NOT_FOUND_RESPONSE,
)
def get_prediction(
    prediction_id: int,
    repository: Annotated[PredictionRepository, Depends(get_repository)],
) -> PredictionRecord:
    try:
        return repository.get(prediction_id)
    except PredictionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Predicción {prediction_id} no encontrada",
        ) from exc


@router.put(
    "/predictions/{prediction_id}",
    response_model=PredictionRecord,
    summary="Actualizar una predicción",
    description=(
        "Reemplaza las medidas, vuelve a ejecutar el modelo y actualiza el "
        "registro almacenado."
    ),
    responses=_NOT_FOUND_RESPONSE,
)
def update_prediction(
    prediction_id: int,
    payload: PredictionCreate,
    model: Annotated[IrisModel, Depends(get_model)],
    repository: Annotated[PredictionRepository, Depends(get_repository)],
) -> PredictionRecord:
    predicted_class, predicted_label, confidence = model.predict(_features_list(payload))
    try:
        return repository.update(
            prediction_id=prediction_id,
            features=payload.model_dump(),
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence=confidence,
        )
    except PredictionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Predicción {prediction_id} no encontrada",
        ) from exc


@router.delete(
    "/predictions/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar una predicción",
    responses=_NOT_FOUND_RESPONSE,
)
def delete_prediction(
    prediction_id: int,
    repository: Annotated[PredictionRepository, Depends(get_repository)],
) -> None:
    try:
        repository.delete(prediction_id)
    except PredictionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Predicción {prediction_id} no encontrada",
        ) from exc
