import numpy as np
from numpy.random import SeedSequence
from goodpoints import compressc, kt, compress
from typing import Optional, Union
import math

def largest_power_of_four(n: int) -> int:
    """
    Returns the largest power of four less than or equal to `n`.

    Args:
        n (int): Input number. Must be positive.

    Returns:
        int: Largest power of four ≤ n.

    Raises:
        ValueError: If `n` is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    return 4**( (n.bit_length() - 1 )//2)

def compute_m(n: int, target_size: int, g: int, num_bins: int) -> int:
    """
    Calculate the number of halving rounds needed in the Kernel Thinning algorithm
    to reduce the coreset to the desired target size.

    Args:
        n (int): Total number of samples.
        target_size (int): Desired size of the coreset.
        g (int): Oversampling parameter for `compress_kt()`.
        num_bins (int): Number of bins for `compress_kt()`.

    Returns:
        int: Number of halving rounds required to reach `target_size`.

    Raises:
        ValueError: If `target_size` can not be reached with set ``n, `num_bins` and `g`.
    """
    n_prime = num_bins * largest_power_of_four(math.floor(n / num_bins))
    compress_coreset_size = min(2**g * math.sqrt(n_prime * num_bins), n_prime)
    if target_size > compress_coreset_size:
        raise ValueError("target_size is too large for thinx_compress with set num_bins and g parameters")
    if target_size == compress_coreset_size:
        return 0
    else:
        total_halvings = 1
        while compress_coreset_size / (2**total_halvings) > target_size:
            total_halvings += 1
        return total_halvings

def thinx_compress(
        X: np.ndarray,
        kernel_type: bytes,
        k_params: Union[np.ndarray, float] = np.ones(1),
        g: int = 4,
        num_bins: int = 32,
        target_size: Optional[int] = None,
        delta: float = 0.5,
        seed: Optional[int] = None,
        mean0: bool = False
    ) -> np.ndarray:
    """
    Compresses the input data X into a coreset of size `target_size` using kernel thinning.
    Behaves identically to `compresspp_kt()` when target_size = sqrt(nearest power of 4 ≤ n).

    Args:
        X (np.ndarray): Input array of samples.
        kernel_type (bytes): Kernel identifier (b"gaussian", b"sobolev",
            b"ineverse_multiquadric" or b"matern").
        k_params (np.ndarray): Kernel parameters.
        g (int): Oversampling parameter for `compress_kt()`.
        num_bins (int): Number of bins for `compress_kt()`.
        target_size (int): Desired size of the coreset. Must be a power of 2.
        delta (float): Failure probability parameter for kernel thinning.
        seed (int): Random seed for reproducibility.
        mean0 (bool): If False, final KT call minimizes MMD to empirical measure over
            the input points. Otherwise minimizes MMD to the 0 measure; this
            is useful when the kernel has expectation zero under a target measure.

    Returns:
        np.ndarray: Array of row indices into X representing the coreset.

    Raises:
        ValueError: If `target_size` is out of bounds or not a power of 2.
    """
    n = X.shape[0]
    if n == 0:
        raise ValueError("Input X cannot be empty")
    if target_size is None:
        target_size = int(np.sqrt(largest_power_of_four(n)))
    if target_size < 1 or target_size > n:
        raise ValueError(f"target_size must be in [1, {n}] and must be a power of 2.")
    if not (target_size > 0 and (target_size & (target_size - 1) == 0)):
        raise ValueError(f"target_size ({target_size}) must be a power of 2.")

    nearest_pow_four = largest_power_of_four(n)
    sqrt_nearest_pow_four = int(math.sqrt(nearest_pow_four))

    # --- If the n is not a power of 4, thin down to the nearest ---
    # --- power of 4 using standard thinning (i.e., by retaining every t-th index) ---
    if nearest_pow_four != n:
        input_indices = np.linspace(n-1, 0, nearest_pow_four, dtype=int)[::-1]
        return input_indices[ thinx_compress(
            X[input_indices], kernel_type, k_params=k_params, g=g, 
            num_bins=num_bins, target_size=target_size, delta=delta, seed=seed, mean0=mean0) ]

    # --- Calculate the number of halving rounds ---
    m = compute_m(n, target_size, g, num_bins)
    log2_target_size = int(math.log2(target_size))

    # --- Align with compresspp_kt() function for `target_size` == sqrt_nearest_pow_four ---
    # --- Directly thin to target_size if no compress step is needed ---
    if sqrt_nearest_pow_four == target_size:
        if log2_target_size <= m:
            K = np.empty((n, n))
            compressc.compute_K(X, np.arange(n, dtype=int), kernel_type, k_params, K)
            return kt.thin_K(K, K, log2_target_size, delta=delta, seed=seed, mean0=mean0)

    # --- Otherwise, split delta between compress and thin steps ---
    thin_frac = m / (m + (2**m) * (log2_target_size - m))
    thin_delta = delta * thin_frac
    compress_delta = delta * (1-thin_frac)

    # --- Generate one seed for the Compress step and one for the final Thin step ---
    seed_seqs = SeedSequence(seed).spawn(2)
    compress_seed = seed_seqs[0].generate_state(1)
    thin_seed = seed_seqs[1].generate_state(1)

    #
    # --- Compress step ---
    #
    # --- Break input into num_bins bins, use Compress(g) to create a coreset of size ---
    # --- 2^g sqrt(n / num_bins) for each bin, and concatenate bin coresets ---
    compress_coreset = compress.compress_kt(
        X, kernel_type, g=g, num_bins=num_bins, k_params=k_params, 
        delta=compress_delta, seed=compress_seed)

    #
    # --- Thin step ---
    #
    # --- Allocate memory for kernel matrix ---
    compress_coreset_size = compress_coreset.shape[0]
    K = np.empty((compress_coreset_size, compress_coreset_size))
    # --- Compute compress_coreset kernel matrix in place ---
    compressc.compute_K(X, compress_coreset, kernel_type, k_params, K)

    # --- Use kt.thin to reduce coreset size from 2^g * sqrt(n * num_bins) to target_size ---
    return compress_coreset[ 
        kt.thin_K(K, K, m, delta=thin_delta, seed=thin_seed, mean0=mean0)]
