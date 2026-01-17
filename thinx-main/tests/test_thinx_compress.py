import pytest
import numpy as np
from thinx.thinx_compress import thinx_compress
from goodpoints import compressc, kt, compress


# --- Mock functions to avoid heavy computation ---
def fake_compress_kt(X, kernel_type, g=4, num_bins=32, k_params=np.ones(1), delta=0.5, seed=None):
    """Return sequential indices for compress step"""
    n = X.shape[0]
    n_per_bin = n // num_bins
    nearest_pow = 4 ** ((n_per_bin.bit_length() - 1) // 2)
    n_prime = nearest_pow * num_bins
    coreset_size = min(n_prime, int(np.sqrt(n_prime * num_bins) * (2 ** g)))
    return np.arange(min(coreset_size, n), dtype=int)


def fake_thin_K(K_split, K_swap, m, delta=0.5, seed=None, unique=False, mean0=False):
    """Return first floor(n/2^m) indices"""
    n = K_split.shape[0]
    if m == 0:
        return np.arange(n)
    size = max(1, n // (2 ** m))
    return np.arange(size)


def fake_compute_K(X, idx, kernel_type, k_params, K):
    """Fill kernel matrix with ones"""
    K[:, :] = 1.0


# --- Tests ---
@pytest.mark.parametrize("n,target_size", [
    (1024, 1), (1024, 2), (1024, 4), (1024, 8), (1024, 16), (1024, 32), (1024, 64), (1024, 128), (1024, 256), (1024, 512),
])
def test_thinx_compress_various_target_sizes(monkeypatch, n, target_size):
    X = np.random.randn(n, 5)

    # Patch heavy functions
    monkeypatch.setattr(compress, "compress_kt", fake_compress_kt)
    monkeypatch.setattr(kt, "thin_K", fake_thin_K)
    monkeypatch.setattr(compressc, "compute_K", fake_compute_K)

    res = thinx_compress(X, kernel_type=b"gaussian", target_size=target_size)
    assert res.ndim == 1
    assert len(res) == target_size
    assert np.all(res >= 0) and np.all(res < n)
    assert len(np.unique(res)) == len(res)


def test_bonxai_compress_default_target_size(monkeypatch):
    X = np.random.randn(65, 3)

    monkeypatch.setattr(compress, "compress_kt", fake_compress_kt)
    monkeypatch.setattr(kt, "thin_K", fake_thin_K)
    monkeypatch.setattr(compressc, "compute_K", fake_compute_K)

    res = thinx_compress(X, kernel_type=b"gaussian")
    expected_size = int(np.sqrt(64))  # largest_power_of_four(65)=64 -> sqrt=8
    assert len(res) == expected_size


def test_thinx_compress_invalid_target_size(monkeypatch):
    X = np.random.randn(16, 3)

    monkeypatch.setattr(compress, "compress_kt", fake_compress_kt)
    monkeypatch.setattr(kt, "thin_K", fake_thin_K)
    monkeypatch.setattr(compressc, "compute_K", fake_compute_K)

    with pytest.raises(ValueError):
        thinx_compress(X, kernel_type=b"gaussian", target_size=0)

    with pytest.raises(ValueError):
        thinx_compress(X, kernel_type=b"gaussian", target_size=17)


def test_thinx_compress_empty_input():
    X = np.empty((0, 3))
    with pytest.raises(ValueError):
        thinx_compress(X, kernel_type=b"gaussian")
