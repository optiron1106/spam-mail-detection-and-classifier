from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACTS_DIR / "logreg_model.joblib"
DATASET_PATH = PROJECT_ROOT / "mail_data.csv"


@dataclass
class TrainedArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression


class SpamModelService:
    def __init__(self, dataset_path: Path, artifacts_dir: Path) -> None:
        self.dataset_path: Path = dataset_path
        self.artifacts_dir: Path = artifacts_dir
        self._artifacts: Optional[TrainedArtifacts] = None

    def ensure_artifacts(self) -> None:
        if self._artifacts is None:
            self._artifacts = self._load_or_train()

    def predict(self, text: str) -> Dict:
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Input text must be a non-empty string")
        self.ensure_artifacts()
        assert self._artifacts is not None

        vectorizer = self._artifacts.vectorizer
        model = self._artifacts.model

        features = vectorizer.transform([text])
        predicted_label_array = model.predict(features)
        predicted_label_id: int = int(predicted_label_array[0])

        # Map probabilities; original training maps: spam -> 0, ham -> 1
        class_order = list(model.classes_)
        probas = model.predict_proba(features)[0]
        ham_index = class_order.index(1)
        ham_probability = float(probas[ham_index])
        spam_probability = float(1.0 - ham_probability)

        label_text = "ham" if predicted_label_id == 1 else "spam"

        return {
            "label": label_text,
            "label_id": predicted_label_id,
            "probabilities": {"spam": spam_probability, "ham": ham_probability},
        }

    # Internal helpers

    def _load_or_train(self) -> TrainedArtifacts:
        # Try load
        if VECTORIZER_PATH.exists() and MODEL_PATH.exists():
            vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
            model: LogisticRegression = joblib.load(MODEL_PATH)
            return TrainedArtifacts(vectorizer=vectorizer, model=model)
        # Else train and save
        artifacts = self._train_and_save()
        return artifacts

    def _train_and_save(self) -> TrainedArtifacts:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. Place 'mail_data.csv' at project root."
            )
        raw_mail_data = pd.read_csv(self.dataset_path)
        mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), "")

        # Map categories: spam -> 0, ham -> 1 (consistent with original script)
        mail_data.loc[mail_data["Category"] == "spam", "Category"] = 0
        mail_data.loc[mail_data["Category"] == "ham", "Category"] = 1

        texts = mail_data["Message"]
        labels = mail_data["Category"].astype("int")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=3
        )

        vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_features, y_train)

        # Quick report to stdout for visibility
        train_pred = model.predict(X_train_features)
        test_pred = model.predict(X_test_features)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(model, MODEL_PATH)

        return TrainedArtifacts(vectorizer=vectorizer, model=model)


_service_singleton: Optional[SpamModelService] = None


def get_service() -> SpamModelService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = SpamModelService(
            dataset_path=DATASET_PATH, artifacts_dir=ARTIFACTS_DIR
        )
        _service_singleton.ensure_artifacts()
    return _service_singleton
