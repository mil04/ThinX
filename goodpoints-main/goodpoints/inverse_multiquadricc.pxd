"""Cython declarations for Inverse Multiquadric kernel functionality used by other files
"""
cdef double inverse_multiquadric_kernel_two_points(const double[:] X1,
                                                    const double[:] X2,
                                                    const double[:] c_params) noexcept nogil

cdef double inverse_multiquadric_kernel_one_point(const double[:] X1,
                                                  const double[:] c_params) noexcept nogil

