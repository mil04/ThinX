import numpy as np
import pytest
from thinx.evaluation import Evaluator

@pytest.fixture
def ground_truth():
    gt_exp = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]])
    gt_pts = np.array([[1, 2], [3, 4], [5, 6]])
    return gt_exp, gt_pts

def test_evaluate_explanation_basic(monkeypatch, ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)

    monkeypatch.setattr("thinx.evaluation.compute_mae", lambda a, b: 0.123)
    monkeypatch.setattr("thinx.evaluation.top_k_score", lambda a, b, k=5: 0.9)

    explanation = np.array([[0.1, 0.1, 0.3],
                            [0.3, 0.5, 0.7]])

    result = ev.evaluate_explanation(explanation, time_elapsed=2.5, num_samples=42)
    assert set(result.keys()) == {"mae", "top_k", "explanation_time", "size"}
    assert result["mae"] == pytest.approx(0.123)
    assert result["top_k"] == pytest.approx(0.9)
    assert result["explanation_time"] == 2.5
    assert result["size"] == 42

def test_evaluate_explanation_integration_with_real_metrics():
    gt_exp = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6, 0.7, 0.8],
    ])
    gt_pts = np.array([[1, 2], [3, 4], [5, 6]]) 
    ev = Evaluator(gt_exp, gt_pts)

    explanation = gt_exp + 0.1  # small shift
    result = ev.evaluate_explanation(explanation, time_elapsed=1.0, num_samples=2)
    assert isinstance(result["mae"], float)
    assert isinstance(result["top_k"], float)
    assert isinstance(result["explanation_time"], float)
    assert result["size"] == 2

def test_evaluate_compression_basic(monkeypatch, ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)

    monkeypatch.setattr("thinx.evaluation.compute_mmd", lambda a, b: 0.456)
    compressed = np.array([[1, 2], [3, 4]])

    result = ev.evaluate_compression(compressed)
    assert set(result.keys()) == {"mmd"}
    assert result["mmd"] == pytest.approx(0.456)

def test_evaluate_compression_integration(ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)
    compressed = np.array([[1, 2], [5, 6]])
    result = ev.evaluate_compression(compressed)

    assert "mmd" in result
    assert isinstance(result["mmd"], float)
    assert result["mmd"] >= 0.0

def test_evaluate_explanation_shape_mismatch_raises(ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)
    # mismatch in shape
    bad = np.array([[0.1, 0.2]])  # 1x2 not 2x3
    with pytest.raises(ValueError, match="explanation.*shape"):
        ev.evaluate_explanation(bad, time_elapsed=0.1, num_samples=5)

def test_evaluate_explanation_negative_time_or_size_raises(ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)
    with pytest.raises(ValueError):
        ev.evaluate_explanation(gt_exp, time_elapsed=-0.1, num_samples=1)
    with pytest.raises(ValueError):
        ev.evaluate_explanation(gt_exp, time_elapsed=0.1, num_samples=-5)

@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_evaluate_explanation_rejects_non_finite(ground_truth, val):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)
    bad = gt_exp.copy()
    bad[0, 0] = val
    with pytest.raises(ValueError):
        ev.evaluate_explanation(bad, time_elapsed=0.1, num_samples=1)

def test_evaluate_compression_dim_mismatch_raises(ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)
    bad = np.c_[np.array([[1, 2]]), np.array([[9]])] 
    with pytest.raises(ValueError):
        ev.evaluate_compression(bad)

def test_evaluate_compression_idempotent_zero(ground_truth):
    gt_exp, gt_pts = ground_truth
    ev = Evaluator(gt_exp, gt_pts)
    out = ev.evaluate_compression(gt_pts.copy())
    assert out["mmd"] == pytest.approx(0.0, abs=1e-12)
