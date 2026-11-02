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


cdef _get_buffer(key, shape, dtype=float):
    """A zeroed buffer of the given shape, reused across calls in this thread."""
    cdef object buffers = getattr(_local, "buffers", None)
    cdef object buf
    if buffers is None:
        buffers = _local.buffers = {}
    buf = buffers.get(key)
    if buf is None or buf.shape != shape or buf.dtype != np.dtype(dtype):
        buf = np.zeros(shape, dtype=dtype)
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

    for d in range(n_terms):
        # The coefficients occupy the leading half of the circulant embedding;
        # asking for a longer transform zero-pads the rest, so no padded input
        # buffer is needed
        fft_w_coeffs = scipy.fft.rfft(
            w[:, d], n=n_fft_coeffs, workers=num_threads
        )
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

    # A two-dimensional transform is a pass along the rows followed by a pass
    # along the columns, and taking the two passes separately lets both of them
    # skip work that a single `rfft2`/`irfft2` cannot know about:
    #
    #   - the coefficients occupy only the top left quadrant of the circulant
    #     embedding, so the forward row pass has half its rows zero;
    #   - only the bottom right quadrant of the result is ever read, so the
    #     inverse row pass is needed for half the rows.
    #
    # Together that removes about a quarter of the transform work. The column
    # spectrum is kept between calls, and since only its top half is ever
    # written, the zeroed bottom half stays zeroed.
    spectrum = _get_buffer(
        ("2d", n_fft_coeffs),
        (n_fft_coeffs, n_fft_coeffs // 2 + 1),
        dtype=complex,
    )

    for d in range(n_terms):
        spectrum[:n_interpolation_points_1d] = scipy.fft.rfft(
            w[:, d].reshape(n_interpolation_points_1d, n_interpolation_points_1d),
            n=n_fft_coeffs,
            axis=1,
            workers=num_threads,
        )

        fft_w_coefficients = scipy.fft.fft(spectrum, axis=0, workers=num_threads)
        fft_w_coefficients *= fft_kernel_tilde

        columns = scipy.fft.ifft(
            fft_w_coefficients, axis=0, overwrite_x=True, workers=num_threads
        )
        fft_out_buffer = scipy.fft.irfft(
            columns[n_interpolation_points_1d:],
            n=n_fft_coeffs,
            axis=1,
            workers=num_threads,
        )

        o[:, d] = fft_out_buffer[:, n_interpolation_points_1d:].ravel()
