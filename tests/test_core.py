import numpy as np
import pytest

from example_repo.core import fit_logreg, predict_proba


def test_fit_logreg_runs_and_returns_reasonable_metrics():
    res = fit_logreg(n_samples=300, n_features=8, noise=0.6, test_size=0.3, seed=42)
    assert 0.0 <= res.train_accuracy <= 1.0
    assert 0.0 <= res.test_accuracy <= 1.0
    assert res.n_train + res.n_test == 300


def test_predict_proba_shape_and_range():
    res = fit_logreg(n_samples=200, n_features=5, noise=0.4, test_size=0.2, seed=0)
    X = np.random.randn(7, 5)
    p = predict_proba(res.model, X)
    assert p.shape == (7,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        fit_logreg(test_size=1.0)