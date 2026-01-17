import numpy as np
import pytest
from thinx.metrics import (
    compute_mae,
    compute_mmd,
    top_k_score,
    topk_pair_overlap,
)

def test_compute_mae_basic():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.5, 1.0, 2.0])
    # manual: (0 + 0.5 + 1 + 1) / 4 = 0.625
    assert compute_mae(a, b) == pytest.approx(0.625)

def test_compute_mae_zero_when_equal():
    x = np.array([1.2, -3.4, 5.6])
    assert compute_mae(x, x) == pytest.approx(0.0)

def test_compute_mae_raises_on_shape_mismatch():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        compute_mae(a, b)

def test_compute_mmd_nonnegative_and_zero_same_samples():
    # Same distribution -> MMD^2 should be ~0
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 4))
    val_same = compute_mmd(X, X, gamma=None)
    assert val_same >= -1e-12 
    assert val_same == pytest.approx(0.0, abs=1e-10)

def test_compute_mmd_positive_for_different_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0, scale=1.0, size=(12, 5))
    Y = rng.normal(loc=1.0, scale=1.0, size=(10, 5))  
    val = compute_mmd(X, Y, gamma=None)
    assert val >= 0.0
    assert val > 0.0

def test_compute_mmd_raises_on_feature_mismatch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    Y = rng.normal(size=(7, 3)) 
    with pytest.raises(ValueError):
        compute_mmd(X, Y, gamma=None)

def test_top_k_score_1d_simple():
    exp = np.array([0.0, 0.2, 0.8, 0.5])
    gt  = np.array([0.1, 0.8, 0.6, 0.4])
    # k=2: exp top2 -> indices of largest two (by abs): {2,3}
    #      gt  top2 -> {1,2}
    # overlap = {2} -> 1/2 = 0.5
    score = top_k_score(exp, gt, k=2)
    assert score == pytest.approx(0.5)

def test_top_k_score_2d_per_sample_average():
    exp = np.array([
        [0.0, 0.9, 0.1, 0.2],   # top2 -> {1,3}
        [0.5, 0.4, 0.3, 0.2],   # top2 -> {0,1}
    ])
    gt = np.array([
        [0.1, 0.8, 0.3, 0.7],   # top2 -> {1,3} -> overlap 2/2 = 1.0
        [0.9, 0.1, 0.8, 0.0],   # top2 -> {0,2} -> overlap {0} -> 1/2 = 0.5
    ])
    score = top_k_score(exp, gt, k=2)
    assert score == pytest.approx((1.0 + 0.5) / 2.0)

def test_top_k_score_3d_multiclass_sums_last_axis():
    exp = np.array([
        [
            [0.0, 0.0],   # f0 sum=0.0
            [0.5, 0.5],   # f1 sum=1.0
            [0.2, 0.1],   # f2 sum=0.3
            [0.3, 0.9],   # f3 sum=1.2 (top)
        ],
        [
            [0.4, 0.3],   # f0 sum=0.7
            [0.1, 0.1],   # f1 sum=0.2
            [0.0, 0.0],   # f2 sum=0.0
            [0.6, 0.4],   # f3 sum=1.0 (top)
        ],
    ])
    gt = np.array([
        [
            [0.0, 0.0],
            [0.4, 0.6],
            [0.1, 0.1],
            [0.8, 0.5],
        ],
        [
            [0.5, 0.3],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.7, 0.3],
        ],
    ])
    # For k=2:
    # sample 1: exp top2 = {3,1}, gt top2 = {3,1} -> 1.0
    # sample 2: exp top2 = {3,0}, gt top2 = {3,0} -> 1.0
    score = top_k_score(exp, gt, k=2)
    assert score == pytest.approx(1.0)

def test_top_k_score_handles_k_bigger_than_features():
    exp = np.array([0.1, 0.2, 0.3])
    gt  = np.array([0.3, 0.2, 0.1])
    score = top_k_score(exp, gt, k=10)
    assert 0.0 <= score <= 1.0

def test_topk_pair_overlap_identical_is_one():
    A = np.array([[[0.0, 1.0],
                   [1.0, 0.0]]]) 
    B = A.copy()
    assert topk_pair_overlap(A, B, k=1) == pytest.approx(1.0)

def test_topk_pair_overlap_basic_overlap_with_k_clip():
    A = np.array([[[0.0, 0.9, 0.1],
                   [0.9, 0.0, 0.8],
                   [0.1, 0.8, 0.0]]]) 
    B = np.array([[[0.0, 0.8, 0.7],
                   [0.8, 0.0, 0.2],
                   [0.7, 0.2, 0.0]]]) 
    # Upper values (i<j) for A: [0.9, 0.1, 0.8] -> ranks: 0.9, 0.8, 0.1
    # For B:                         [0.8, 0.7, 0.2] -> ranks: 0.8, 0.7, 0.2
    # k=5 gets clipped to m=3 internally
    score = topk_pair_overlap(A, B, k=5)
    # Top sets:
    #   A top3 -> {pairs for 0.9, 0.8, 0.1} = all three
    #   B top3 -> all three
    # denom = min(len(topA), len(topB)) = 3; intersection size = 3; => 1.0
    assert score == pytest.approx(1.0)

def test_compute_mae_shape_mismatch_raises():
    a = np.zeros((2, 3))
    b = np.zeros((2, 2))
    with pytest.raises(ValueError):
        compute_mae(a, b)

@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_compute_mae_rejects_non_finite(val):
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    b[0] = val
    with pytest.raises(ValueError):
        compute_mae(a, b)

def test_mmd_identity_zero():
    X = np.random.RandomState(0).randn(20, 3)
    m = compute_mmd(X, X)
    assert m == pytest.approx(0.0, abs=1e-10)

def test_mmd_symmetric_and_non_negative():
    rng = np.random.RandomState(1)
    X = rng.randn(30, 2)
    Y = rng.randn(25, 2) + 0.5
    mxy = compute_mmd(X, Y)
    myx = compute_mmd(Y, X)
    assert mxy >= 0.0
    assert mxy == pytest.approx(myx, rel=1e-12, abs=1e-12)

def test_mmd_dim_mismatch_raises():
    X = np.zeros((10, 3))
    Y = np.zeros((8, 4))
    with pytest.raises(ValueError):
        compute_mmd(X, Y)

def test_top_k_score_ties_deterministic():
    gt = np.array([[1.0, 1.0, 0.0]])
    ex = np.array([[1.0, 1.0, 0.0]])
    s1 = top_k_score(ex, gt, k=1)
    s2 = top_k_score(ex, gt, k=2)
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0

@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_top_k_score_rejects_non_finite(val):
    gt = np.array([[0.1, 0.2, 0.3]])
    ex = gt.copy()
    ex[0, 0] = val
    with pytest.raises(ValueError):
        top_k_score(ex, gt, k=1)

def test_topk_pair_overlap_dim_mismatch_raises():
    A = np.zeros((2, 3, 3))
    B = np.zeros((2, 4, 4))
    with pytest.raises(ValueError):
        topk_pair_overlap(A, B, k=1)

@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_topk_pair_overlap_rejects_non_finite(val):
    A = np.zeros((1, 3, 3)); B = np.zeros((1, 3, 3))
    A = A.copy(); A[0, 0, 1] = val
    with pytest.raises(ValueError):
        topk_pair_overlap(A, B, k=1)

def test_topk_pair_overlap_d_small_policy():
    A = np.zeros((1, 1, 1))
    B = np.zeros((1, 1, 1))
    with pytest.raises(ValueError):
        topk_pair_overlap(A, B, k=1)
