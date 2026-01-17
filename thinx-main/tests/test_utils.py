import numpy as np
import pytest
import torch
import os
import random
from thinx import utils


def test_benchmark_lists_are_valid():
    assert len(utils.CTR23_ALL) > 0
    assert len(utils.CC18_ALL) > 0
    assert all(isinstance(x, int) for x in utils.CTR23_ALL)
    assert all(isinstance(x, int) for x in utils.CC18_ALL)

def test_set_global_seed_affects_all_generators():
    utils.set_global_seed(123)
    a_np1 = np.random.rand(3)
    a_t1 = torch.rand(3)
    a_py1 = [random.random() for _ in range(3)]
    hashseed1 = os.environ.get("PYTHONHASHSEED")

    utils.set_global_seed(123)
    a_np2 = np.random.rand(3)
    a_t2 = torch.rand(3)
    a_py2 = [random.random() for _ in range(3)]
    hashseed2 = os.environ.get("PYTHONHASHSEED")

    assert np.allclose(a_np1, a_np2)
    assert torch.allclose(a_t1, a_t2)
    assert a_py1 == a_py2
    assert hashseed1 == hashseed2  # ideally '123'

def test_set_global_seed_different_seed_changes_streams():
    utils.set_global_seed(1)
    x1 = np.random.rand(5)
    utils.set_global_seed(2)
    x2 = np.random.rand(5)
    assert not np.allclose(x1, x2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_global_seed_cuda_safe():
    utils.set_global_seed(7)
    x = torch.rand(3, device="cuda")
    assert x.is_cuda

def test_possible_g_values_basic():
    g_vals = utils.possible_g_values(n_samples=512, num_bins=4)
    # should return a list of decreasing integers (reversed)
    assert isinstance(g_vals, list)
    assert all(isinstance(g, int) for g in g_vals)
    assert g_vals == sorted(g_vals, reverse=True)
    # at least one element expected
    assert len(g_vals) > 0

def test_possible_g_values_monotonic_change_with_samples():
    g_small = utils.possible_g_values(64, 4)
    g_large = utils.possible_g_values(1024, 4)
    assert len(g_large) >= len(g_small)

def test_possible_num_bins_values_small_n():
    vals = utils.possible_num_bins_values(100)
    # 4, 16, 64 are <= 100 -> expected
    assert vals == [4, 16, 64]

def test_possible_num_bins_values_increases_monotonically():
    vals = utils.possible_num_bins_values(500)
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    # last value is power of 4 and <= n_samples
    assert 4 ** int(np.log(vals[-1]) / np.log(4)) == vals[-1]
    assert vals[-1] <= 500

@pytest.mark.parametrize("n, expected", [
    (1, 1),    # 4**0 -> 1 -> sqrt(1) = 1
    (2, 1),
    (3, 1),
    (4, 2),    # 4**1 -> 4 -> sqrt(4) = 2
    (5, 2),
    (15, 2),
    (16, 4),   # 4**2 -> 16 -> sqrt(16) = 4
    (17, 4),
    (63, 4),
    (64, 8),   # 4**3 -> 64 -> sqrt(64) = 8
    (65, 8),
])
def test_compresspp_kt_output_size_various(n, expected):
    X = np.arange(n)
    assert utils.compresspp_kt_output_size(X) == expected

def test_compresspp_kt_output_size_zero_len_raises():
    with pytest.raises((ValueError, ZeroDivisionError, FloatingPointError)):
        utils.compresspp_kt_output_size(np.arange(0))
