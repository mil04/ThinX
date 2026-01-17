import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double inverse_multiquadric_kernel_two_points(const double[:] X1,
                                                    const double[:] X2,
                                                    const double[:] c_params) noexcept nogil:
    """
    Computes a sum of inverse multiquadric kernels:
    k(X1, X2) = sum_j 1 / sqrt(c_j + ||X1 - X2||^2)

    Args:
      X1: array of size d
      X2: array of size d
      c_params: array of size num_kernels, each c_j > 0 
    """
    cdef long d = X1.shape[0]
    cdef long num_kernels = c_params.shape[0]
    
    cdef double sq_dist = 0.
    cdef double diff
    cdef long j
    
    for j in range(d):
        diff = X1[j] - X2[j]
        sq_dist += diff * diff

    cdef double kernel_sum = 0.
    for j in range(num_kernels):
        kernel_sum += 1.0 / sqrt(c_params[j] + sq_dist)
    
    return kernel_sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double inverse_multiquadric_kernel_one_point(const double[:] X1,
                                                  const double[:] c_params) noexcept nogil:
    """
    Computes the sum of inverse multiquadric kernels between X1 and itself:
    k(X1, X1) = sum_j 1 / sqrt(c_j)

    Args:
      X1: array of size d (ignored, kept for consistency)
      c_params: array of kernel parameters, each > 0
    """
    cdef long num_kernels = c_params.shape[0]
    cdef long j
    cdef double kernel_sum = 0.

    for j in range(num_kernels):
        kernel_sum += 1.0 / sqrt(c_params[j])

    return kernel_sum

