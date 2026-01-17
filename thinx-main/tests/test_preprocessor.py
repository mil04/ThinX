import numpy as np
import pytest
import torch
from thinx.preprocessor import Compressor, Preprocessor

class DummyModel:
    def __init__(self):
        self.called = False
    def predict_proba(self, X):
        self.called = True
        X = np.asarray(X)
        s = X.sum(axis=1, keepdims=True)
        p0 = 1 / (1 + np.exp(s))
        p1 = 1 - p0
        return np.hstack([p0, p1])
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

@pytest.fixture
def small_data():
    X = np.random.randn(10, 3)
    y = np.random.randint(0, 2, size=10)
    model = DummyModel()
    return X, y, model

def test_kernel_thinning(monkeypatch, small_data):
    X, y, model = small_data
    indices = np.arange(5)
    monkeypatch.setattr("thinx.preprocessor.thinx_compress", lambda *a, **kw: indices)
    comp = Compressor(X, y, model)
    Xr, yr, idx, t = comp._kernel_thinning(1, 2, 3, 0.5, b"gaussian", np.ones(2))
    assert np.allclose(Xr, X[idx])
    assert np.allclose(yr, y[idx])
    assert np.all(idx == indices)
    assert isinstance(t, float)

def test_stein_thinning_gaussian(monkeypatch, small_data):
    X, y, model = small_data
    monkeypatch.setattr("thinx.preprocessor.thin", lambda X, grad, m: np.arange(min(len(X), 4)))
    comp = Compressor(X, y, model)
    Xr, yr, idx, t = comp._stein_thinning(4, b"gaussian")
    assert Xr.shape[1] == X.shape[1]
    assert len(idx) == 4
    assert isinstance(t, float)

def test_stein_thinning_kde(monkeypatch, small_data):
    X, y, model = small_data
    monkeypatch.setattr("thinx.preprocessor.thin", lambda X, grad, m: np.arange(3))
    comp = Compressor(X, y, model)
    Xr, yr, idx, t = comp._stein_thinning(3, b"kde")
    assert Xr.shape == (3, X.shape[1])

def test_stein_thinning_gmm(monkeypatch, small_data):
    X, y, model = small_data
    class FakeGMM:
        def __init__(self, n_components, covariance_type): pass
        def fit(self, X): pass
        @property
        def means_(self): return np.zeros((2, X.shape[1]))
        @property
        def covariances_(self): return np.stack([np.eye(X.shape[1])]*2)
        def predict_proba(self, X): return np.full((len(X), 2), 0.5)
    monkeypatch.setattr("thinx.preprocessor.GaussianMixture", FakeGMM)
    monkeypatch.setattr("thinx.preprocessor.thin", lambda X, grad, m: np.arange(5))
    comp = Compressor(X, y, model)
    Xr, yr, idx, t = comp._stein_thinning(5, b"gmm")
    assert Xr.shape == (5, X.shape[1])

def test_stein_thinning_invalid_grad_type_raises(monkeypatch, small_data):
    X, y, model = small_data
    comp = Compressor(X, y, model)
    with pytest.raises(ValueError):
        comp._stein_thinning(3, b"unknown")

def test_influence_compression(monkeypatch, small_data):
    X, y, model = small_data
    class DummyTorchModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.lin = torch.nn.Linear(X.shape[1], 2)
        def forward(self, x): return self.lin(x)
    torch_model = DummyTorchModel()
    class FakeInf:
        def __init__(self, *a, **kw): pass
        def fit(self, loader): return self
        def influences(self, Xt, yt, Xtr, ytr, mode="up"):
            return torch.ones((len(Xt), len(Xtr)))
    monkeypatch.setattr("thinx.preprocessor.CgInfluence", FakeInf)
    comp = Compressor(X, y, torch_model)
    Xr, yr, idx, t, mat = comp._influence_compression(target_size=4)
    assert Xr.shape[0] == 4
    assert isinstance(t, float)
    assert isinstance(mat, torch.Tensor)

def test_arfpy_compression(monkeypatch, small_data):
    X, y, model = small_data
    target_size = 5
    class FakeARFInstance:
        def __init__(self, **kwargs): pass
        def forde(self): return None
        def forge(self, n):
            import pandas as pd
            d = {f"feat_{i}": np.random.randn(n) for i in range(X.shape[1])}
            d["label"] = np.random.randint(0, 2, n)
            return pd.DataFrame(d)
    monkeypatch.setattr("thinx.preprocessor.arf_mod.arf", FakeARFInstance)
    comp = Compressor(X, y, model)
    Xr, yr, idx, t = comp._arfpy_compression(target_size=target_size)
    assert Xr.shape == (target_size, X.shape[1])
    assert yr.shape[0] == target_size
    assert np.all(idx == -1)
    assert isinstance(t, float)

def test_iid_sampling(small_data):
    X, y, model = small_data
    comp = Compressor(X, y, model)
    Xr, yr, idx, t = comp._iid_sampling(5)
    assert Xr.shape[0] == 5
    assert len(idx) == 5
    assert isinstance(t, float)

def test_data_with_predictions(small_data):
    X, y, model = small_data
    pre = Preprocessor(X, y, model=model, compression_method="iid")
    Xaug, preds = pre._data_with_predictions()
    assert Xaug.shape[1] == X.shape[1] + preds.shape[1]
    assert model.called

def test_dispatch_calls_correct_method(monkeypatch, small_data):
    X, y, model = small_data
    monkeypatch.setattr("thinx.preprocessor.resolve_kernel_params", lambda k, X: (b"gaussian", np.ones(2)))
    comp = Preprocessor(X, y, model=model, compression_method="iid")
    res = comp._dispatch_compression(X, y, 0, 0, 5, "gaussian", 0.5, None)
    assert len(res) == 4

@pytest.mark.parametrize("method", ["kernel_thinning", "stein_thinning", "influence", "arfpy", "iid"])
def test_dispatch_each_method(monkeypatch, small_data, method):
    X, y, model = small_data
    monkeypatch.setattr(
        "thinx.preprocessor.resolve_kernel_params",
        lambda name, X, seed: (b"gaussian", np.ones(2))
    )
    pre = Preprocessor(X, y, model=model, compression_method=method)
    comp = Compressor(X, y, model=model)
    for name in [
        "_kernel_thinning",
        "_stein_thinning",
        "_influence_compression",
        "_arfpy_compression",
        "_iid_sampling",
    ]:
        monkeypatch.setattr(Compressor, name, lambda *a, **kw: (X[:2], y[:2], np.array([0, 1]), 0.1))
    res = pre._dispatch_compression(X, y, 4, 4, 4 , "gaussian", 0.5, "gaussian")
    assert isinstance(res, tuple)
    assert len(res) == 4

def test_dispatch_unknown_method_raises(small_data):
    X, y, model = small_data
    with pytest.raises(ValueError):
        Preprocessor(
            X, y,
            model=model,
            compression_method="nope"
        )

def test_preprocess_none(monkeypatch, small_data):
    X, y, model = small_data
    monkeypatch.setattr("thinx.preprocessor.resolve_kernel_params", lambda k, X: (b"gaussian", np.ones(2)))
    pre = Preprocessor(X, y, model=model, compression_method="iid", data_modification_method="none")
    monkeypatch.setattr(Preprocessor, "_dispatch_compression", lambda *a, **kw: (X[:2], y[:2], np.array([0, 1]), 0.2))
    Xr, yr, idx, t = pre.preprocess()
    assert Xr.shape[0] == 2
    assert isinstance(t, float)

def test_preprocess_predictions(monkeypatch, small_data):
    X, y, model = small_data
    monkeypatch.setattr("thinx.preprocessor.resolve_kernel_params", lambda k, X: (b"gaussian", np.ones(2)))
    pre = Preprocessor(X, y, model=model, compression_method="iid", data_modification_method="predictions")
    monkeypatch.setattr(Preprocessor, "_dispatch_compression", lambda *a, **kw: (X[:2], y[:2], np.array([0, 1]), 0.2))
    Xr, yr, idx, t = pre.preprocess()
    assert Xr.shape[0] == 2

def test_preprocess_stratified(monkeypatch, small_data):
    X, y, model = small_data
    monkeypatch.setattr(
        Preprocessor,
        "_dispatch_compression",
        lambda *a, **kw: (X[:1], y[:1], np.array([0]), 0.1)
    )
    pre = Preprocessor(X, y, model=model, compression_method="iid", data_modification_method="stratified")
    monkeypatch.setattr(Preprocessor, "_dispatch_compression", lambda *a, **kw: (X[:2], y[:2], np.array([0, 1]), 0.1))
    Xr, yr, idx, t = pre.preprocess(target_size=4)
    assert len(Xr) > 0
    assert isinstance(t, float)

def test_preprocess_invalid_data_mod_method(small_data):
    X, y, model = small_data
    with pytest.raises(ValueError):
        Preprocessor(
            X, y,
            model=model,
            compression_method="iid",
            data_modification_method="weird"
        )

def test_predictions_mode_calls_predict_proba(monkeypatch, small_data):
    X, y, model = small_data
    called = {"proba": False}
    def fake_predict_proba(Xinp):
        called["proba"] = True
        return np.column_stack([np.full(len(Xinp), 0.3), np.full(len(Xinp), 0.7)])
    model.predict_proba = fake_predict_proba

    monkeypatch.setattr(
        Preprocessor, "_dispatch_compression",
        lambda *a, **kw: (a[1][:2], a[2][:2], np.array([0,1]), 0.02)
    )
    pre = Preprocessor(X, y, model=model, compression_method="iid", data_modification_method="predictions")
    Xr, yr, idx, t = pre.preprocess()

    assert called["proba"]
    assert Xr.shape[1] == X.shape[1] + 2

def test_dispatch_returns_matrix_when_available(monkeypatch, small_data):
    X, y, model = small_data
    pre = Preprocessor(X, y, model=model, compression_method="influence")
    monkeypatch.setattr(
        Compressor, "_influence_compression",
        lambda *a, **kw: (X[:2], y[:2], np.array([0,1]), 0.1, np.ones((2, len(X))))
    )
    res = pre._dispatch_compression(X, y, 0, 0, 10, "gaussian", 0.5, None)
    assert len(res) == 5

def test_iid_sampling_respects_seed(small_data):
    X, y, model = small_data
    comp1 = Compressor(X, y, model=model, seed=123)
    comp2 = Compressor(X, y, model=model, seed=123)
    X1, y1, idx1, _ = comp1._iid_sampling(target_size=4)
    X2, y2, idx2, _ = comp2._iid_sampling(target_size=4)
    assert np.array_equal(idx1, idx2)
