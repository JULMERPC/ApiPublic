"""JSON-file persistence for prediction records.

The repository keeps the whole dataset in memory while handling a request and
writes it back atomically. A process-wide lock serializes read-modify-write
cycles so concurrent requests do not lose data.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas import PredictionRecord


class PredictionNotFoundError(KeyError):
    """Raised when a prediction id does not exist."""


class PredictionRepository:
    def __init__(self, data_file: Path) -> None:
        self.data_file = Path(data_file)
        self._lock = threading.Lock()
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self._write({"next_id": 1, "items": []})

    def _read(self) -> dict[str, Any]:
        with self.data_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.setdefault("next_id", 1)
        data.setdefault("items", [])
        return data

    def _write(self, data: dict[str, Any]) -> None:
        temp_file = self.data_file.with_suffix(self.data_file.suffix + ".tmp")
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_file, self.data_file)

    def create(
        self,
        features: dict[str, float],
        predicted_class: int,
        predicted_label: str,
        confidence: float,
    ) -> PredictionRecord:
        with self._lock:
            data = self._read()
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "id": data["next_id"],
                **features,
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "created_at": now,
                "updated_at": now,
            }
            data["items"].append(record)
            data["next_id"] += 1
            self._write(data)

        return PredictionRecord(**record)

    def list_predictions(self, skip: int = 0, limit: int = 50) -> list[PredictionRecord]:
        with self._lock:
            data = self._read()

        items = sorted(data["items"], key=lambda item: item["id"], reverse=True)
        return [PredictionRecord(**item) for item in items[skip : skip + limit]]

    def get(self, prediction_id: int) -> PredictionRecord:
        with self._lock:
            data = self._read()

        for item in data["items"]:
            if item["id"] == prediction_id:
                return PredictionRecord(**item)
        raise PredictionNotFoundError(prediction_id)

    def update(
        self,
        prediction_id: int,
        features: dict[str, float],
        predicted_class: int,
        predicted_label: str,
        confidence: float,
    ) -> PredictionRecord:
        with self._lock:
            data = self._read()
            for item in data["items"]:
                if item["id"] == prediction_id:
                    item.update(features)
                    item["predicted_class"] = predicted_class
                    item["predicted_label"] = predicted_label
                    item["confidence"] = confidence
                    item["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._write(data)
                    return PredictionRecord(**item)
        raise PredictionNotFoundError(prediction_id)

    def delete(self, prediction_id: int) -> None:
        with self._lock:
            data = self._read()
            remaining = [item for item in data["items"] if item["id"] != prediction_id]
            if len(remaining) == len(data["items"]):
                raise PredictionNotFoundError(prediction_id)
            data["items"] = remaining
            self._write(data)
