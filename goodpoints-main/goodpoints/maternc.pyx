import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double matern_kernel_two_points(const double[:] X1,
                                     const double[:] X2,
                                     const double[:] params) noexcept nogil:
    """
    Compute sum of Matérn kernels between two points.
    params = flattened array [sigma1, ell1, nu1, sigma2, ell2, nu2, ...]
    Only nu = 0.5, 1.5, 2.5 are supported in nogil mode
    """
    cdef long d = X1.shape[0]
    cdef long num_kernels = params.shape[0] // 3
    cdef long j, k
    cdef double sq_dist = 0.
    cdef double r, diff, sigma, ell, nu, kernel_val
    cdef double kernel_sum = 0.

    # Compute squared Euclidean distance
    for k in range(d):
        diff = X1[k] - X2[k]
        sq_dist += diff * diff
    r = sqrt(sq_dist)

    # Sum over all kernels
    for j in range(num_kernels):
        sigma = params[3*j + 0]
        ell   = params[3*j + 1]
        nu    = params[3*j + 2]

        if r == 0.0:
            kernel_val = sigma * sigma
        elif nu == 0.5:
            kernel_val = sigma * sigma * exp(-r / ell)
        elif nu == 1.5:
            kernel_val = sigma * sigma * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
        elif nu == 2.5:
            kernel_val = sigma * sigma * (1 + sqrt(5)*r/ell + 5*r*r/(3*ell*ell)) * exp(-sqrt(5)*r/ell)
        else:
            # Unsupported nu in nogil mode
            assert 0, "Matérn kernel: nu not supported in nogil mode (use 0.5, 1.5, 2.5)."

        kernel_sum += kernel_val

    return kernel_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double matern_kernel_one_point(const double[:] X1,
                                    const double[:] params) noexcept nogil:
    """
    Compute diagonal element k(X1,X1) = sum_j sigma_j^2
    params = flattened array [sigma1, ell1, nu1, sigma2, ell2, nu2, ...]
    """
    cdef long num_kernels = params.shape[0] // 3
    cdef long j
    cdef double sigma, kernel_sum = 0.

    for j in range(num_kernels):
        sigma = params[3*j + 0]
        kernel_sum += sigma * sigma

    return kernel_sum
