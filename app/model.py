"""Iris model training, persistence and inference."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IrisModel:
    """Small wrapper around a scikit-learn pipeline.

    The model is trained once and persisted with joblib so later startups load
    it from disk instead of retraining.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.pipeline: Pipeline | None = None
        self.target_names: list[str] = []
        self.test_accuracy: float | None = None

    def load_or_train(self) -> None:
        """Load the persisted model or train a new one when missing."""
        if self.model_path.exists():
            payload = joblib.load(self.model_path)
            self.pipeline = payload["pipeline"]
            self.target_names = list(payload["target_names"])
            self.test_accuracy = payload.get("test_accuracy")
            logger.info("Loaded Iris model from %s", self.model_path)
            return

        self._train()

    def _train(self) -> None:
        dataset = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.2,
            random_state=42,
            stratify=dataset.target,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=200)),
            ]
        )
        pipeline.fit(x_train, y_train)

        self.pipeline = pipeline
        self.target_names = list(dataset.target_names)
        self.test_accuracy = float(pipeline.score(x_test, y_test))

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "target_names": self.target_names,
                "test_accuracy": self.test_accuracy,
            },
            self.model_path,
        )
        logger.info(
            "Trained Iris model (test accuracy=%.3f) and saved it to %s",
            self.test_accuracy,
            self.model_path,
        )

    def predict(self, features: list[float]) -> tuple[int, str, float]:
        """Return (class index, label, confidence) for one sample."""
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded. Call load_or_train() first.")

        probabilities = self.pipeline.predict_proba([features])[0]
        class_index = int(probabilities.argmax())
        return (
            class_index,
            self.target_names[class_index],
            float(probabilities[class_index]),
        )
