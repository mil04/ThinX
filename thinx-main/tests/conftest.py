import numpy as np
import pytest
rng = np.random.default_rng(42)

@pytest.fixture
def toy_vectors():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.5, 1.0, 2.0])
    return a, b

@pytest.fixture
def toy_matrices():
    X = rng.normal(size=(6, 4))
    Y = rng.normal(size=(5, 4))
    return X, Y