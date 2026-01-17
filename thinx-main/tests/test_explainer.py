import numpy as np
import pytest
import torch
from thinx.explainer import Explainer
from thinx.pytorch_nn import PyTorchNN, BasicNeuralNetwork

class DummyClsModel:
    def predict_proba(self, X):
        X = np.asarray(X)
        s = X.sum(axis=1, keepdims=True)
        p0 = 1 / (1 + np.exp(s))
        p1 = 1 - p0
        return np.hstack([p0, p1])

class DummyRegModel:
    def predict(self, X):
        X = np.asarray(X)
        return X.sum(axis=1)

def test_init_rejects_unsupported_explainer():
    with pytest.raises(ValueError):
        Explainer(DummyClsModel(), explainer_name="nope", task_type="classification")

def test_init_rejects_unsupported_strategy():
    with pytest.raises(ValueError):
        Explainer(DummyClsModel(), explainer_name="shap", task_type="classification", strategy="weird")

def test_init_requires_correct_model_interface():
    with pytest.raises(ValueError):
        Explainer(DummyRegModel(), explainer_name="shap", task_type="classification")
    with pytest.raises(ValueError):
        Explainer(DummyClsModel(), explainer_name="shap", task_type="regression")

def test_explain_requires_labels_for_sage_and_influence():
    Xf = np.random.randn(5, 3)
    Xb = np.random.randn(6, 3)
    e = Explainer(DummyClsModel(), explainer_name="sage", task_type="classification", strategy="kernel")
    with pytest.raises(ValueError):
        e.explain(Xf, Xb, n_jobs=1, y_foreground=None)
    e2 = Explainer(DummyClsModel(), explainer_name="influence", task_type="classification")
    with pytest.raises(ValueError):
        e2.explain(Xf, Xb, n_jobs=1, y_foreground=np.zeros(5, int), y_background=None)

def test_explain_shap_kernel_classification(monkeypatch):
    Xb = np.random.randn(8, 4)
    Xf = np.random.randn(6, 4)
    class FakeShapOutput:
        def __init__(self, values):
            self.values = values
    class FakeKernelExplainer:
        def __init__(self, f, background, seed=None):
            self.f = f
        def __call__(self, X, silent=True):
            return FakeShapOutput(np.ones((len(X), X.shape[1], 2)) * 0.5)
    monkeypatch.setattr("thinx.explainer.shap", type("SHAP", (), {
        "KernelExplainer": FakeKernelExplainer,
        "maskers": type("M", (), {"Independent": lambda *a, **k: None}),
        "PermutationExplainer": None,
    }))
    e = Explainer(DummyClsModel(), explainer_name="shap", task_type="classification", strategy="kernel", seed=0)
    vals, elapsed = e._explain_shap(X_background=Xb, X_foreground=Xf, n_jobs=1, verbose=False)
    assert vals.shape == (len(Xf), Xf.shape[1])
    assert isinstance(elapsed, float)

def test_explain_shap_permutation_batched(monkeypatch):
    Xb = np.random.randn(10, 3)
    Xf = np.random.randn(15, 3)
    class FakeShapValues:
        def __init__(self, X):
            self.values = np.full((len(X), X.shape[1]), 0.42)
    class FakePermutationExplainer:
        def __init__(self, f, masker, seed=None):
            pass
        def __call__(self, X, silent=True):
            return FakeShapValues(np.asarray(X))
    class FakeMaskers:
        @staticmethod
        def Independent(X, max_samples=None):
            return None
    monkeypatch.setattr("thinx.explainer.shap", type("SHAP", (), {
        "PermutationExplainer": FakePermutationExplainer,
        "maskers": FakeMaskers,
        "KernelExplainer": None,
    }))
    e = Explainer(DummyRegModel(), explainer_name="shap", task_type="regression", strategy="permutation", seed=0)
    vals, elapsed = e._explain_shap(X_background=Xb, X_foreground=Xf, n_jobs=2)
    assert vals.shape == (len(Xf), Xf.shape[1])
    assert np.allclose(vals, 0.42)
    assert isinstance(elapsed, float)

def test_explain_sage_kernel_and_permutation(monkeypatch):
    Xb = np.random.randn(7, 4)
    Xf = np.random.randn(6, 4)
    y = np.array([0, 1, 1, 0, 1, 0])
    class FakeImputer:
        def __init__(self, f, data): pass
    class FakeEstimator:
        def __init__(self, imputer, loss, random_state=None, n_jobs=None): pass
        def __call__(self, X, y, bar=False, verbose=False):
            class Obj: pass
            o = Obj()
            o.values = np.full((len(X), X.shape[1]), 0.5)
            return o
    monkeypatch.setattr("thinx.explainer.sage", type("SAGE", (), {
        "MarginalImputer": FakeImputer,
        "KernelEstimator": FakeEstimator,
        "PermutationEstimator": FakeEstimator
    }))
    e1 = Explainer(DummyClsModel(), "sage", "classification", strategy="kernel", seed=0)
    vals1, t1 = e1._explain_sage(X_background=Xb, X_foreground=Xf, y_foreground=y, n_jobs=4)
    assert vals1.shape == (len(Xf), Xf.shape[1])
    assert isinstance(t1, float)
    e2 = Explainer(DummyClsModel(), "sage", "classification", strategy="permutation", seed=0)
    vals2, t2 = e2._explain_sage(X_background=Xb, X_foreground=Xf, y_foreground=y, n_jobs=2)
    assert vals2.shape == (len(Xf), Xf.shape[1])
    assert isinstance(t2, float)

def test_explain_sage_requires_labels():
    Xb = np.random.randn(5, 3)
    Xf = np.random.randn(4, 3)
    e = Explainer(DummyClsModel(), "sage", "classification", strategy="kernel")
    with pytest.raises(ValueError):
        e.explain(Xf, Xb, n_jobs=1, y_foreground=None)

def test_explain_shapiq_returns_list_and_time(monkeypatch):
    Xb = np.random.randn(6, 3)
    Xf = np.random.randn(5, 3)
    class FakeImputer:
        def __init__(self, model, data, sample_size): pass
    class FakeIV:
        def __init__(self, d):
            self.d = d
        def get_n_order_values(self, n):
            if n == 1:
                return np.ones(self.d)
            elif n == 2:
                return np.ones((self.d, self.d))
            else:
                return None
    class FakeExplainer:
        def __init__(self, model, data, approximator, index, max_order, imputer): pass
        def explain(self, x, budget, random_state):
            return FakeIV(len(x))
    monkeypatch.setattr(
        "thinx.explainer.shapiq",
        type("S", (), {
            "MarginalImputer": FakeImputer,
            "TabularExplainer": FakeExplainer
        })
    )
    e = Explainer(DummyRegModel(), "shapiq", "regression")
    pairwise_list, elapsed = e._explain_shapiq(X_background=Xb, X_foreground=Xf, n_jobs=1, verbose=False)
    assert isinstance(pairwise_list, tuple) or isinstance(pairwise_list, list)
    assert len(pairwise_list) == len(Xf)
    for p in pairwise_list:
        assert isinstance(p, np.ndarray)
        assert p.shape == (Xf.shape[1], Xf.shape[1])
    assert isinstance(elapsed, float)

def test_expected_gradients_requires_pytorchann():
    Xb = np.random.randn(5, 3)
    Xf = np.random.randn(4, 3)
    e = Explainer(DummyClsModel(), "expected_gradients", "classification")
    with pytest.raises(TypeError):
        e._explain_expected_gradients(Xb, Xf, n_jobs=2)

def test_expected_gradients_returns_tensor_and_time(monkeypatch):
    ann = PyTorchNN(task_type="classification", epochs=1, batch_size=2, random_state=0)
    ann.model_ = BasicNeuralNetwork(n_inputs=3, n_outputs=2)
    Xb = np.random.randn(6, 3)
    Xf = np.random.randn(5, 3)
    def fake_predict_proba(X):
        X = np.asarray(X)
        s = X.sum(axis=1, keepdims=True)
        p0 = 1 / (1 + np.exp(s))
        p1 = 1 - p0
        return np.hstack([p0, p1])
    ann.predict_proba = fake_predict_proba
    class FakeIG:
        def __init__(self, model): pass
        def attribute(self, inputs, baselines, target=None):
            return torch.ones_like(inputs)
    class FakeParallel:
        def __init__(self, n_jobs=None): pass
        def __call__(self, tasks):
            return [t() if callable(getattr(t, "__call__", None)) else t for t in tasks]
    def delayed(fn):
        class C:
            def __init__(self, fn, args, kwargs):
                self.fn = fn; self.args = args; self.kw = kwargs
            def __call__(self):
                return self.fn(*self.args, **self.kw)
        def wrapper(*args, **kwargs):
            return C(fn, args, kwargs)
        return wrapper
    monkeypatch.setattr("thinx.explainer.captum", type("C", (), {"attr": type("A", (), {"IntegratedGradients": FakeIG})}))
    monkeypatch.setattr("thinx.explainer.joblib", type("J", (), {"Parallel": FakeParallel, "delayed": delayed}))
    e = Explainer(ann, "expected_gradients", "classification")
    vals, elapsed = e._explain_expected_gradients(X_background=Xb, X_foreground=Xf, n_jobs=2)
    assert isinstance(vals, torch.Tensor)
    assert vals.shape == (len(Xf), Xf.shape[1])
    assert isinstance(elapsed, float)

def test_influence_with_torch_module(monkeypatch):
    class TinyNet(torch.nn.Module):
        def __init__(self, d_in=3, d_out=2):
            super().__init__()
            self.lin = torch.nn.Linear(d_in, d_out)
        def forward(self, x):
            return self.lin(x)
        def predict_proba(self, X):
            out = self.forward(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(out, dim=1)
            return probs.detach().numpy()
    Xb = np.random.randn(8, 3)
    yb = np.random.randint(0, 2, size=8)
    Xf = np.random.randn(5, 3)
    yf = np.random.randint(0, 2, size=5)
    class FakeInf:
        def __init__(self, model, loss_fn, regularization, rtol, atol, solve_simultaneously): pass
        def fit(self, loader): return self
        def influences(self, X_test, y_test, X_train, y_train, mode="up"):
            return torch.zeros((len(X_test), len(X_train)))
    monkeypatch.setattr("thinx.explainer.CgInfluence", FakeInf)
    e = Explainer(TinyNet(), "influence", "classification")
    mat, elapsed = e._explain_influence(X_background=Xb, y_background=yb, X_foreground=Xf, y_foreground=yf)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (len(Xf), len(Xb))
    assert isinstance(elapsed, float)

def test_influence_rejects_non_torch_model():
    e = Explainer(DummyRegModel(), "influence", "regression")
    with pytest.raises(TypeError):
        e._explain_influence(
            X_background=np.random.randn(4, 2),
            y_background=np.zeros(4, int),
            X_foreground=np.random.randn(3, 2),
            y_foreground=np.zeros(3, int),
        )
