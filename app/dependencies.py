"""FastAPI dependencies resolved from application state."""

from fastapi import Request

from app.model import IrisModel
from app.repository import PredictionRepository


def get_repository(request: Request) -> PredictionRepository:
    return request.app.state.repository


def get_model(request: Request) -> IrisModel:
    return request.app.state.model
