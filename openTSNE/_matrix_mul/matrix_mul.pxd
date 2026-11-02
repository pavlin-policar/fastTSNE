# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3


cdef void matrix_multiply_fft_1d(
    double[::1] kernel_tilde,
    double[:, ::1] w_coefficients,
    double[:, ::1] out,
    Py_ssize_t num_threads=*,
)

cdef void matrix_multiply_fft_2d(
    double[:, ::1] kernel_tilde,
    double[:, ::1] w_coefficients,
    double[:, ::1] out,
    Py_ssize_t num_threads=*,
)
