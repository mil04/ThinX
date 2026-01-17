"""Cython declarations for Matérn kernel functionality"""

cdef double matern_kernel_two_points(const double[:] X1,
                                     const double[:] X2,
                                     const double[:] params) noexcept nogil

cdef double matern_kernel_one_point(const double[:] X1,
                                    const double[:] params) noexcept nogil
