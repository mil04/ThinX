import numpy as np
import pytest
import torch
from thinx.pytorch_nn import BasicNeuralNetwork, PyTorchNN

def test_basic_network_forward_shape():
    n_in, n_out = 4, 3
    net = BasicNeuralNetwork(n_in, n_out)
    x = torch.randn(5, n_in)
    y = net(x)
    assert y.shape == (5, n_out)

def _toy_classification(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y

def _toy_multiclass(n=45, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    s = X @ rng.normal(size=d)
    y = np.digitize(s, np.quantile(s, [1/3, 2/3]))  # 0,1,2
    return X, y

def _toy_regression(n=40, d=3, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (X @ w + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y

def test_ann_classification_fit_predict_and_proba_shapes():
    X, y = _toy_classification(n=60, d=6)
    model = PyTorchNN(task_type="classification", epochs=5, lr=1e-3, batch_size=16, random_state=42)
    model.fit(X, y)

    # predict
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})

    # predict_proba
    probs = model.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    # probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

def test_ann_multiclass_output_dim_matches_unique_y():
    X, y = _toy_multiclass()
    n_classes = len(np.unique(y))
    model = PyTorchNN(task_type="classification", epochs=5, lr=1e-3, batch_size=16, random_state=7)
    model.fit(X, y)

    probs = model.predict_proba(X[:8])
    assert probs.shape == (8, n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

def test_ann_regression_fit_predict_shape_and_dtype():
    X, y = _toy_regression()
    model = PyTorchNN(task_type="regression", epochs=5, lr=1e-3, batch_size=16, random_state=11)
    model.fit(X, y)

    preds = model.predict(X[:7])
    assert preds.shape == (7,)
    assert np.issubdtype(preds.dtype, np.floating)

def test_predict_before_fit_raises():
    X, _ = _toy_classification(n=10, d=3)
    model = PyTorchNN(task_type="classification", epochs=1)
    with pytest.raises(RuntimeError):
        model.predict(X)

def test_predict_proba_not_available_for_regression():
    X, y = _toy_regression()
    model = PyTorchNN(task_type="regression", epochs=1)
    model.fit(X, y)
    with pytest.raises(AttributeError):
        model.predict_proba(X)

def test_invalid_task_type_raises_on_fit():
    X, y = _toy_classification()
    model = PyTorchNN(task_type="unknown", epochs=1)
    with pytest.raises(ValueError):
        model.fit(X, y)

@pytest.fixture
def cls_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    Xtr, Xte = X[:80], X[80:]
    ytr, yte = y[:80], y[80:]
    return Xtr, ytr, Xte, yte

@pytest.fixture
def reg_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 6))
    y = 2.0 * X[:, 0] - 0.7 * X[:, 1] + 0.1 * rng.normal(size=len(X))
    Xtr, Xte = X[:100], X[100:]
    ytr, yte = y[:100], y[100:]
    return Xtr, ytr, Xte, yte

def test_classification_improves_over_chance(cls_data):
    Xtr, ytr, Xte, yte = cls_data
    ann = PyTorchNN(
        task_type="classification",
        epochs=20
    )
    ann.fit(Xtr, ytr)
    preds = ann.predict(Xte)
    acc = (preds == yte).mean()
    assert acc >= 0.6 

def test_predict_requires_fit(cls_data):
    Xtr, ytr, Xte, _ = cls_data
    ann = PyTorchNN(task_type="classification", epochs=1)
    with pytest.raises((AttributeError, RuntimeError, ValueError)):
        ann.predict(Xte)
    ann.fit(Xtr, ytr)
    _ = ann.predict(Xte)

def test_fit_predict_regression_shapes(reg_data):
    Xtr, ytr, Xte, yte = reg_data
    ann = PyTorchNN(task_type="regression", epochs=8)
    ann.fit(Xtr, ytr)
    preds = ann.predict(Xte)
    assert preds.shape == (len(Xte),)
    assert np.isfinite(preds).all()

def test_regression_mse_reasonable(reg_data):
    Xtr, ytr, Xte, yte = reg_data
    ann = PyTorchNN(task_type="regression", epochs=25)
    ann.fit(Xtr, ytr)
    preds = ann.predict(Xte)
    mse = np.mean((preds - yte) ** 2)
    assert mse < np.var(yte)
