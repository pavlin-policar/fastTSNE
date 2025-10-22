# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()

from .quad_tree cimport QuadTree


ctypedef fused sparse_index_type:
    cnp.int32_t
    cnp.int64_t


cpdef double[:, ::1] compute_gaussian_perplexity(
    double[:, :] distances,
    double[:] desired_perplexities,
    double perplexity_tol=*,
    Py_ssize_t max_iter=*,
    Py_ssize_t num_threads=*,
)

cpdef tuple estimate_positive_gradient_nn(
    sparse_index_type[:] indices,
    sparse_index_type[:] indptr,
    double[:] P_data,
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    double dof=*,
    Py_ssize_t num_threads=*,
    bint should_eval_error=*,
)

cpdef tuple estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=*,
    double dof=*,
    Py_ssize_t num_threads=*,
    bint pairwise_normalization=*,
)

cpdef double estimate_negative_gradient_fft_1d(
    double[::1] embedding,
    double[::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
    double dof=*,
    Py_ssize_t num_threads=*,
)

cpdef tuple prepare_negative_gradient_fft_interpolation_grid_1d(
    double[::1] reference_embedding,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
    double dof=*,
    double padding=*,
    Py_ssize_t num_threads=*,
)

cpdef double estimate_negative_gradient_fft_1d_with_grid(
    double[::1] embedding,
    double[::1] gradient,
    double[:, ::1] y_tilde_values,
    double[::1] box_lower_bounds,
    Py_ssize_t n_interpolation_points,
    double dof,
)

cpdef double estimate_negative_gradient_fft_2d(
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
    double dof=*,
    Py_ssize_t num_threads=*,
)

cpdef tuple prepare_negative_gradient_fft_interpolation_grid_2d(
    double[:, ::1] reference_embedding,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
    double dof=*,
    double padding=*,
    Py_ssize_t num_threads=*,
)

cpdef double estimate_negative_gradient_fft_2d_with_grid(
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double[:, ::1] y_tilde_values,
    double[::1] box_x_lower_bounds,
    double[::1] box_y_lower_bounds,
    Py_ssize_t n_interpolation_points,
    double dof,
)
