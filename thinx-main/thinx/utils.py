import os
import random
import math
import torch
import numpy as np
from scipy.spatial.distance import pdist

# CTR23 benchmark suite - regression tasks - used for experiments
CTR23_LARGE = [44964, 44975, 44981, 44992]
CTR23_SMALL = [44956, 44963, 44969, 44971, 44973, 44974, 44976, 44977, 44978, 44979, 44980, 44983, 44984, 44989, 44990, 44993, 45012]
CTR23_ALL = sorted(CTR23_SMALL + CTR23_LARGE)

# CC18 benchmark suite - classification tasks - used for experiments
CC18_LARGE = [28, 44, 182, 300, 554, 1486, 1475, 4538, 1478, 40499, 40668, 40996, 40923, 40927]
CC18_SMALL = [6, 32, 151, 1053, 1590, 1489, 1497, 4534, 1461, 40983, 41027, 23517, 40701]
CC18_ALL = sorted(CC18_LARGE + CC18_SMALL)


def set_global_seed(seed: int) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JOBLIB_START_METHOD"] = "spawn"


def possible_g_values(n_samples: int, num_bins: int) -> list[int]:
    """
    Find all valid values of g for the compress_kt() function to be used.

    Args:
        n_samples (int): Total number of samples.
        num_bins (int): Number of bins used.

    Returns:
        list[int]: Valid g values.
    """
    x = n_samples // num_bins
    power = 1
    while power * 4 <= x:
        power *= 4
    n_prime = num_bins * power

    g = 0
    possible_g = []
    while (2 ** g) * math.sqrt(n_prime * num_bins) <= n_prime:
        possible_g.append(g)
        g += 1
    return list(reversed(possible_g))


def possible_num_bins_values(n_samples: int) -> list[int]:
    """
    Find valid values of num_bins for the compress_kt() function to be used.

    Args:
        n_samples (int): Total number of samples.

    Returns:
        list[int]: All powers of 4 (4^k) that do not exceed n_samples.
    """
    possible_num_bins = []
    power = 1 # assume that num_bins starts at 4
    while (val := 4 ** power) <= n_samples:
        possible_num_bins.append(val)
        power += 1
    return possible_num_bins


def compresspp_kt_output_size(X: np.ndarray) -> int:
    """
    Find the size of the coreset returned by the compresspp_kt() function.

    Args:
        X (np.ndarray): Input array of all samples

    Returns:
        int: coreset size after compression in compresspp_kt() function.

    Raises:
        ValueError: If `X` is empty.
    """
    n = len(X)
    if n <= 0:
        raise ValueError("X must be non-empty")
    n_prime = 4 ** int(np.floor(np.log(n) / np.log(4)))
    return int(np.sqrt(n_prime))


def median_pairwise_distance_sample(
    X: np.ndarray,
    n_pairs: int = 100_000,
    random_state: int | None = None
) -> float:
    """
    Compute or estimate the median pairwise distance between all samples in X.

    Args:
        X (np.ndarray): Input array of all samples.
        n_pairs (int): Number of pairs to sample from all samples.
        random_state (int | None): Seed for reproducibility.

    Returns:
        float: Median pairwise distance.

    Raises:
        ValueError: If `X` is None or empty, or if `n_pairs` <= 0.
    """
    if X is None or len(X) == 0:
        raise ValueError("X cannot be None or empty")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")

    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Compute exact median if the number of pairs is relatively small
    if n * (n - 1) // 2 <= n_pairs:
        return float(np.median(pdist(X)))

    # Sample random pairs to estimate median
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    mask = i != j
    d = np.linalg.norm(X[i[mask]] - X[j[mask]], axis=1)
    return float(np.median(d))