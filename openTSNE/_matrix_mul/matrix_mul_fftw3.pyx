# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import numpy as np
cimport openTSNE._matrix_mul.matrix_mul


cdef extern from 'fftw3.h':
    int fftw_init_threads()
    void fftw_plan_with_nthreads(int)

    cdef unsigned FFTW_ESTIMATE
    cdef unsigned FFTW_DESTROY_INPUT
    ctypedef double fftw_complex[2]

    ctypedef struct _fftw_plan:
       pass

    ctypedef _fftw_plan *fftw_plan

    void fftw_execute(fftw_plan)
    void fftw_destroy_plan(fftw_plan)
    fftw_plan fftw_plan_dft_r2c_1d(int, double*, fftw_complex*, unsigned)
    fftw_plan fftw_plan_dft_c2r_1d(int, fftw_complex*, double*, unsigned)
    fftw_plan fftw_plan_dft_r2c_2d(int, int, double*, fftw_complex*, unsigned)
    fftw_plan fftw_plan_dft_c2r_2d(int, int, fftw_complex*, double*, unsigned)


cdef void matrix_multiply_fft_1d(
    double[::1] kernel_tilde,
    double[:, ::1] w_coefficients,
    double[:, ::1] out,
    Py_ssize_t num_threads=1,
):
    """Multiply the the kernel vectr K tilde with the w coefficients.
    
    Parameters
    ----------
    kernel_tilde : memoryview
        The generating vector of the 2d Toeplitz matrix i.e. the kernel 
        evaluated all all interpolation points from the left most 
        interpolation point, embedded in a circulant matrix (doubled in size 
        from (n_interp, n_interp) to (2 * n_interp, 2 * n_interp) and 
        symmetrized. See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    
    """
    cdef:
        Py_ssize_t n_interpolation_points_1d = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]

        complex[::1] fft_kernel_tilde = np.empty(n_fft_coeffs, dtype=complex)
        complex[::1] fft_w_coeffs = np.empty(n_fft_coeffs, dtype=complex)
        # Note that we can't use the same buffer for the input and output since
        # we only write to the first half of the vector - we'd need to
        # manually zero out the rest of the entries that were inevitably
        # changed during the IDFT, so it's faster to use two buffers, at the
        # cost of some memory
        double[::1] fft_in_buffer = np.zeros(n_fft_coeffs, dtype=float)
        double[::1] fft_out_buffer = np.zeros(n_fft_coeffs, dtype=float)

        Py_ssize_t d, i

    # Compute the FFT of the kernel vector
    cdef fftw_plan plan_dft, plan_idft
    plan_dft = fftw_plan_dft_r2c_1d(
        n_fft_coeffs,
        &kernel_tilde[0], <fftw_complex *>(&fft_kernel_tilde[0]),
        FFTW_ESTIMATE,
    )
    fftw_execute(plan_dft)
    fftw_destroy_plan(plan_dft)

    plan_dft = fftw_plan_dft_r2c_1d(
        n_fft_coeffs,
        &fft_in_buffer[0], <fftw_complex *>(&fft_w_coeffs[0]),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )
    plan_idft = fftw_plan_dft_c2r_1d(
        n_fft_coeffs,
        <fftw_complex *>(&fft_w_coeffs[0]), &fft_out_buffer[0],
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )

    for d in range(n_terms):
        for i in range(n_interpolation_points_1d):
            fft_in_buffer[i] = w_coefficients[i, d]

        fftw_execute(plan_dft)

        # Take the Hadamard product of two complex vectors
        for i in range(n_fft_coeffs):
            fft_w_coeffs[i] = fft_w_coeffs[i] * fft_kernel_tilde[i]

        fftw_execute(plan_idft)

        for i in range(n_interpolation_points_1d):
            # FFTW doesn't perform IDFT normalization, so we have to do it
            # ourselves. This is done by multiplying the result with the number
            #  of points in the input
            out[i, d] = fft_out_buffer[n_interpolation_points_1d + i].real / n_fft_coeffs

    fftw_destroy_plan(plan_dft)
    fftw_destroy_plan(plan_idft)


cdef void matrix_multiply_fft_2d(
    double[:, ::1] kernel_tilde,
    double[:, ::1] w_coefficients,
    double[:, ::1] out,
    Py_ssize_t num_threads=1,
):
    """Multiply the the kernel matrix K tilde with the w coefficients.
    
    Parameters
    ----------
    kernel_tilde : memoryview
        The generating matrix of the 3d Toeplitz tensor i.e. the kernel 
        evaluated all all interpolation points from the top left most 
        interpolation point, embedded in a circulant matrix (doubled in size 
        from (n_interp, n_interp) to (2 * n_interp, 2 * n_interp) and 
        symmetrized. See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    
    """
    cdef:
        Py_ssize_t total_interpolation_points = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t n_interpolation_points_1d = n_fft_coeffs / 2

        fftw_plan plan_dft, plan_idft
        complex[::1] fft_w_coefficients = np.empty(n_fft_coeffs * (n_fft_coeffs / 2 + 1), dtype=complex)
        complex[::1] fft_kernel_tilde = np.empty(n_fft_coeffs * (n_fft_coeffs / 2 + 1), dtype=complex)
        # Note that we can't use the same buffer for the input and output since
        # we only write to the top quadrant of the in matrix - we'd need to
        # manually zero out the rest of the entries that were inevitably
        # changed during the IDFT, so it's faster to use two buffers, at the
        # cost of some memory
        double[:, ::1] fft_in_buffer = np.zeros((n_fft_coeffs, n_fft_coeffs))
        double[:, ::1] fft_out_buffer = np.zeros((n_fft_coeffs, n_fft_coeffs))

        Py_ssize_t d, i, j, idx

    plan_dft = fftw_plan_dft_r2c_2d(
        n_fft_coeffs, n_fft_coeffs,
        &kernel_tilde[0, 0], <fftw_complex *>(&fft_kernel_tilde[0]),
        FFTW_ESTIMATE,
    )
    fftw_execute(plan_dft)
    fftw_destroy_plan(plan_dft)

    plan_dft = fftw_plan_dft_r2c_2d(
        n_fft_coeffs, n_fft_coeffs,
        &fft_in_buffer[0, 0], <fftw_complex *>(&fft_w_coefficients[0]),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )
    plan_idft = fftw_plan_dft_c2r_2d(
        n_fft_coeffs, n_fft_coeffs,
        <fftw_complex *>(&fft_w_coefficients[0]), &fft_out_buffer[0, 0],
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
    )

    for d in range(n_terms):
        for i in range(n_interpolation_points_1d):
            for j in range(n_interpolation_points_1d):
                fft_in_buffer[i, j] = w_coefficients[i * n_interpolation_points_1d + j, d]

        fftw_execute(plan_dft)

        # Take the Hadamard product of two complex vectors
        for i in range(n_fft_coeffs * (n_fft_coeffs / 2 + 1)):
            fft_w_coefficients[i] = fft_w_coefficients[i] * fft_kernel_tilde[i]

        # Invert the computed values at the interpolated nodes
        fftw_execute(plan_idft)
        # FFTW doesn't perform IDFT normalization, so we have to do it
        # ourselves. This is done by multiplying the result with the number of
        # points in the input
        for i in range(n_interpolation_points_1d):
            for j in range(n_interpolation_points_1d):
                idx = i * n_interpolation_points_1d + j
                out[idx, d] = fft_out_buffer[n_interpolation_points_1d + i,
                                                        n_interpolation_points_1d + j] / n_fft_coeffs ** 2

    fftw_destroy_plan(plan_dft)
    fftw_destroy_plan(plan_idft)
