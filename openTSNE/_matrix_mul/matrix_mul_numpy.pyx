# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import threading

import numpy as np
import scipy.fft
cimport openTSNE._matrix_mul.matrix_mul


# The transforms dominate the interpolation-based gradient once the embedding
# has spread out -- at 45k points a converged embedding needs a 1134 x 1134
# grid, and the nine transforms per call are essentially the whole cost. scipy
# exposes pocketfft's own thread pool through `workers`, which parallelizes
# them without needing OpenMP.
#
# The buffers are large (tens of megabytes at that grid size) and the grid size
# changes only when the embedding's extent crosses a box boundary, so they are
# kept between calls. Only the last size is retained; holding on to more would
# waste more memory than the allocations cost. Each thread keeps its own: the
# transforms release the GIL, so embeddings optimized concurrently in different
# threads must not share them.
cdef object _local = threading.local()


cdef _get_buffer(key, shape):
    """A zeroed buffer of the given shape, reused across calls in this thread."""
    cdef object buffers = getattr(_local, "buffers", None)
    cdef object buf
    if buffers is None:
        buffers = _local.buffers = {}
    buf = buffers.get(key)
    if buf is None or buf.shape != shape:
        buf = np.zeros(shape, dtype=float)
        buffers.clear()
        buffers[key] = buf
    return buf


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
    num_threads : Py_ssize_t
        Threads to hand to the FFT backend.

    """
    cdef:
        Py_ssize_t n_interpolation_points_1d = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t d

    if num_threads < 1:
        num_threads = 1

    w = np.asarray(w_coefficients)
    o = np.asarray(out)

    fft_kernel_tilde = scipy.fft.rfft(
        np.asarray(kernel_tilde), workers=num_threads
    )

    # Only the leading half is ever written, so the zeroed tail of a reused
    # buffer stays zeroed
    fft_in_buffer = _get_buffer(("1d", n_fft_coeffs), (n_fft_coeffs,))

    for d in range(n_terms):
        fft_in_buffer[:n_interpolation_points_1d] = w[:, d]

        fft_w_coeffs = scipy.fft.rfft(fft_in_buffer, workers=num_threads)
        fft_w_coeffs *= fft_kernel_tilde
        fft_out_buffer = scipy.fft.irfft(
            fft_w_coeffs, n=n_fft_coeffs, workers=num_threads
        )

        o[:, d] = fft_out_buffer[n_interpolation_points_1d:]


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
    num_threads : Py_ssize_t
        Threads to hand to the FFT backend.

    """
    cdef:
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t n_interpolation_points_1d = n_fft_coeffs // 2
        Py_ssize_t d

    if num_threads < 1:
        num_threads = 1

    w = np.asarray(w_coefficients)
    o = np.asarray(out)

    fft_kernel_tilde = scipy.fft.rfft2(
        np.asarray(kernel_tilde), workers=num_threads
    )

    # Only the top left quadrant is ever written, so the zeroed remainder of a
    # reused buffer stays zeroed
    fft_in_buffer = _get_buffer(
        ("2d", n_fft_coeffs), (n_fft_coeffs, n_fft_coeffs)
    )

    for d in range(n_terms):
        fft_in_buffer[:n_interpolation_points_1d, :n_interpolation_points_1d] = \
            w[:, d].reshape(n_interpolation_points_1d, n_interpolation_points_1d)

        fft_w_coefficients = scipy.fft.rfft2(fft_in_buffer, workers=num_threads)
        fft_w_coefficients *= fft_kernel_tilde
        fft_out_buffer = scipy.fft.irfft2(
            fft_w_coefficients, s=(n_fft_coeffs, n_fft_coeffs), workers=num_threads
        )

        o[:, d] = fft_out_buffer[
            n_interpolation_points_1d:, n_interpolation_points_1d:
        ].ravel()
