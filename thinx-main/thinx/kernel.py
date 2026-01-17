from typing import Tuple
import numpy as np
from thinx.utils import median_pairwise_distance_sample


def resolve_kernel_params(name: str, X: np.ndarray, seed: int) -> Tuple[bytes, np.ndarray]:
    """
    Map a kernel name to its type and parameters (convention for 'goodpoints' package)

    Args:
        name (str): Name of the kernel ("gaussian", "sobolev", "inverse_multiquadric" or "matern")
        X (np.ndarray): Input data used to compute median pairwise distance for some kernels.
        seed (int): Random seed for reproducibility in pairs' sampling for distance computation.

    Returns:
        Tuple[bytes, np.ndarray]:
            - kernel type as bytes,
            - kernel parameters as a NumPy array.

    Raises:
        ValueError: If the kernel name is unknown.
    """
    ell = median_pairwise_distance_sample(X, n_pairs=100_000, random_state=seed)

    if name == "gaussian":
        lam_sqd = ell ** 2
        print(f"Gaussian kernel: lambda^2 = {lam_sqd}")
        return b"gaussian", np.array([lam_sqd], dtype=np.float64)

    elif name == "sobolev":
        k_params = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        print(f"Sobolev kernel: parameters {k_params}")
        return b"sobolev", k_params

    elif name == "inverse_multiquadric":
        print(f"Inverse-Multiquadric kernel: c = {ell ** 2}")
        return b"inverse_multiquadric", np.array([ell ** 2], dtype=np.float64)

    elif name == "matern":
        k_params = np.array([1.0, ell * 0.5, 0.5, 1.0, ell * 1, 1.5, 1.0, ell * 2, 2.5], dtype=np.float64)
        print(f"Matérn kernel: parameters {k_params}")
        return b"matern", k_params

    else:
        raise ValueError(f"Unknown kernel name: {name}")