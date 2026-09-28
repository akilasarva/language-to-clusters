"""Supervised state classifier over extractor features.

A closed-set classifier: it always returns one of the trained labels, so — unlike
the legacy HDBSCAN + nearest-centroid-reassignment path — there is no ``-1``
noise state and 0% of frames come back "unclassified". Two interchangeable
backends behind one interface:

  * ``rf``  — RandomForest (default): no GPU, gives predict_proba for free,
              robust on small feature sets.
  * ``mlp`` — a small torch MLP for when a differentiable head is preferred.

Feature standardization (fit on train) is bundled in, so callers pass raw
extractor features. Save/load round-trips the whole thing (scaler + model +
label vocabulary) via pickle.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class StateClassifier:
    """Standardize -> classify, with a persisted label vocabulary."""

    backend: str = "rf"
    labels: List[str] = field(default_factory=list)   # index -> label string
    n_estimators: int = 200
    mlp_hidden: int = 128
    mlp_epochs: int = 150
    random_state: int = 0
    _model: object = None
    _scaler: object = None
    _mlp_meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, labels: Optional[List[str]] = None):
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if labels is not None:
            self.labels = list(labels)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        if self.backend == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier(
                n_estimators=self.n_estimators, class_weight="balanced",
                random_state=self.random_state, n_jobs=-1)
            self._model.fit(Xs, y)
        elif self.backend == "mlp":
            self._fit_mlp(Xs, y)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        return self

    def _fit_mlp(self, Xs, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        n_classes = int(y.max()) + 1 if len(y) else len(self.labels)
        n_classes = max(n_classes, len(self.labels))
        in_dim = Xs.shape[1]
        model = nn.Sequential(
            nn.Linear(in_dim, self.mlp_hidden), nn.ReLU(),
            nn.Linear(self.mlp_hidden, n_classes))
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
        w = w / w.sum() * (counts > 0).sum() if w.sum() else w
        crit = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.as_tensor(Xs, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.int64)
        model.train()
        for _ in range(self.mlp_epochs):
            opt.zero_grad()
            loss = crit(model(Xt), yt)
            loss.backward()
            opt.step()
        model.eval()
        self._model = model
        self._mlp_meta = {"n_classes": n_classes, "in_dim": in_dim}

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.transform(X)
        if self.backend == "rf":
            proba = self._model.predict_proba(Xs)
            # RF may omit classes unseen in training; expand to full label set.
            full = np.zeros((len(Xs), len(self.labels)), dtype=np.float64)
            for j, cls in enumerate(self._model.classes_):
                full[:, int(cls)] = proba[:, j]
            return full
        else:
            import torch
            with torch.no_grad():
                logits = self._model(torch.as_tensor(Xs, dtype=torch.float32))
                return torch.softmax(logits, dim=1).numpy()

    def label_of(self, idx: int) -> str:
        return self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        blob = {
            "backend": self.backend, "labels": self.labels,
            "n_estimators": self.n_estimators, "scaler": self._scaler,
            "mlp_meta": self._mlp_meta,
        }
        if self.backend == "mlp":
            import torch
            import io
            buf = io.BytesIO()
            torch.save(self._model, buf)
            blob["model_bytes"] = buf.getvalue()
        else:
            blob["model"] = self._model
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "StateClassifier":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        obj = cls(backend=blob["backend"], labels=blob["labels"],
                  n_estimators=blob.get("n_estimators", 200))
        obj._scaler = blob["scaler"]
        obj._mlp_meta = blob.get("mlp_meta", {})
        if blob["backend"] == "mlp":
            import torch
            import io
            obj._model = torch.load(io.BytesIO(blob["model_bytes"]), weights_only=False)
            obj._model.eval()
        else:
            obj._model = blob["model"]
        return obj
