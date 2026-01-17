import numpy as np
import pytest
from unittest.mock import patch
from thinx import kernel


def test_gaussian_kernel_params():
    X = np.zeros((10, 5))
    with patch("thinx.kernel.median_pairwise_distance_sample", return_value=np.sqrt(10)):
        ktype, params = kernel.resolve_kernel_params("gaussian", X=X, seed=42)

    assert ktype == b"gaussian"
    assert np.allclose(params, [10.0])
    assert params.dtype == float


@pytest.mark.parametrize("name,expected_type,expected_params", [
    ("sobolev", b"sobolev", [1.0, 2.0, 3.0]),
    ("inverse_multiquadric", b"inverse_multiquadric", [10.0]),
])
def test_simple_kernels(name, expected_type, expected_params):
    X = np.zeros((8, 3))
    with patch("thinx.kernel.median_pairwise_distance_sample", return_value=np.sqrt(10)):
        ktype, params = kernel.resolve_kernel_params(name, X, seed=42)

    assert ktype == expected_type
    assert np.allclose(params, expected_params)
    assert params.ndim == 1

@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_kernel(nu):
    X = np.zeros((4, 2))
    with patch("thinx.kernel.median_pairwise_distance_sample", return_value=2.0):
        ktype, params = kernel.resolve_kernel_params("matern", X, seed=42)

    assert ktype == b"matern"
    expected_params = np.array([1.0, 2.0*0.5, 0.5, 1.0, 2.0*1, 1.5, 1.0, 2.0*2, 2.5], dtype=np.float64)
    assert np.allclose(params, expected_params)
    assert params.dtype == np.float64

def test_unknown_kernel_raises():
    X = np.zeros((2, 2))
    with pytest.raises(ValueError):
        kernel.resolve_kernel_params("nonexistent", X, 42)


@pytest.mark.parametrize("d", [1, 2, 7, 32])
def test_gaussian_sigma_positive_and_scales_with_features(d):
    X = np.zeros((5, d))
    with patch("thinx.kernel.median_pairwise_distance_sample", return_value=1.0):
        _, p = kernel.resolve_kernel_params("gaussian", X, seed=42)

    assert p.shape == (1,)
    assert p[0] > 0


@pytest.mark.parametrize("name", ["gaussian", "sobolev", "inverse_multiquadric", "matern"])
def test_all_params_are_float64(name):
    X = np.zeros((3, 4))

    if name in ["gaussian", "inverse_multiquadric"]:
        with patch("thinx.kernel.median_pairwise_distance_sample", return_value=1.0):
            _, p = kernel.resolve_kernel_params(name, X, seed=42)
    else:
        _, p = kernel.resolve_kernel_params(name, X, seed=42)

    assert p.dtype == np.float64

@pytest.mark.parametrize("name", ["gaussian","sobolev","inverse_multiquadric"])
def test_all_params_are_float64(name):
    X = np.zeros((3, 4))
    _, p = kernel.resolve_kernel_params(name, X, seed=42)
    assert p.dtype == np.float64
