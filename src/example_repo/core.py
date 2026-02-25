from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    """Set numpy RNG seed for reproducibility."""
    np.random.seed(seed)


@dataclass(frozen=True)
class FitResult:
    """Results returned by fit_logreg()."""
    model: LogisticRegression
    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int


def _make_synthetic_binary_classification(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a simple synthetic binary classification dataset.

    Data generating process:
    - Draw X ~ N(0, I)
    - Create a linear score s = X w + eps
    - y = 1[s > 0]
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=(n_features,))
    eps = noise * rng.normal(size=(n_samples,))
    scores = X @ w + eps
    y = (scores > 0).astype(int)
    return X, y


def fit_logreg(
    *,
    n_samples: int = 500,
    n_features: int = 10,
    noise: float = 0.5,
    test_size: float = 0.25,
    C: float = 1.0,
    seed: int = 0,
    max_iter: int = 500,
) -> FitResult:
    """
    Train a LogisticRegression model on a synthetic dataset and return metrics.

    This is a complete, reproducible example method:
    - data generation
    - train/test split
    - model fitting
    - evaluation
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")
    if n_samples <= 10:
        raise ValueError("n_samples must be > 10.")
    if n_features <= 0:
        raise ValueError("n_features must be > 0.")
    if noise < 0:
        raise ValueError("noise must be >= 0.")
    if C <= 0:
        raise ValueError("C must be > 0.")

    X, y = _make_synthetic_binary_classification(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        seed=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, yhat_train))
    test_acc = float(accuracy_score(y_test, yhat_test))

    return FitResult(
        model=model,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
    )


def predict_proba(
    model: LogisticRegression,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return P(y=1|x) for each row in X.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    proba = model.predict_proba(X)
    # proba[:, 1] is probability of class 1
    return proba[:, 1]