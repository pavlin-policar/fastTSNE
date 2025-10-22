# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import numpy as np
import threading
from cython.parallel import prange, parallel
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free

from .quad_tree cimport QuadTree, Node, is_close
from ._matrix_mul.matrix_mul cimport matrix_multiply_fft_1d, matrix_multiply_fft_2d

from libc.math cimport sqrt, log, pow, exp, fabs, fmax, INFINITY  # Add pow and log
cdef double EPSILON = np.finfo(np.float64).eps


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil
    double fabs(double x) nogil
    double fmax(double x, double y) nogil
    double isinf(long double) nogil
    double INFINITY


cpdef double[:, ::1] compute_gaussian_perplexity(
    double[:, :] distances,
    double[:] desired_perplexities,
    double perplexity_tol=1e-8,
    Py_ssize_t max_iter=200,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t n_samples = distances.shape[0]
        Py_ssize_t n_scales = desired_perplexities.shape[0]
        Py_ssize_t k_neighbors = distances.shape[1]
        double[:, ::1] P = np.zeros_like(distances, dtype=float, order="C")
        double[:, :, ::1] multiscale_P = np.zeros((n_samples, n_scales, k_neighbors))
        double[:, ::1] tau = np.ones((n_samples, n_scales))

        Py_ssize_t i, j, h, iteration
        double[:] desired_entropies = np.log(desired_perplexities)

        double min_tau, max_tau, sum_Pi, sum_PiDj, entropy, entropy_diff, sqrt_tau

    if num_threads < 1:
        num_threads = 1

    for i in prange(n_samples, nogil=True, schedule="guided", num_threads=num_threads):
        min_tau, max_tau = -INFINITY, INFINITY

        # For every scale find a precision tau that fits the perplexity
        for h in range(n_scales):
            for iteration in range(max_iter):
                sum_Pi, sum_PiDj = 0, 0
                sqrt_tau = sqrt(tau[i, h])

                for j in range(k_neighbors):
                    multiscale_P[i, h, j] = sqrt_tau * exp(-distances[i, j] ** 2 * tau[i, h] * 0.5)
                    sum_Pi = sum_Pi + multiscale_P[i, h, j]
                sum_Pi = sum_Pi + EPSILON

                for j in range(k_neighbors):
                    sum_PiDj = sum_PiDj + multiscale_P[i, h, j] / sum_Pi * distances[i, j] ** 2

                entropy = tau[i, h] * 0.5 * sum_PiDj + log(sum_Pi) - log(tau[i, h]) * 0.5
                entropy_diff = entropy - desired_entropies[h]

                if fabs(entropy_diff) <= perplexity_tol:
                    break

                if entropy_diff > 0:
                    min_tau = tau[i, h]
                    if isinf(max_tau):
                        tau[i, h] *= 2
                    else:
                        tau[i, h] = (tau[i, h] + max_tau) * 0.5
                else:
                    max_tau = tau[i, h]
                    if isinf(min_tau):
                        tau[i, h] /= 2
                    else:
                        tau[i, h] = (tau[i, h] + min_tau) * 0.5

        # Get the probability of the mixture of Gaussians with different precisions
        sum_Pi = 0
        for j in range(k_neighbors):
            for h in range(n_scales):
                P[i, j] = P[i, j] + multiscale_P[i, h, j]
                sum_Pi = sum_Pi + multiscale_P[i, h, j]

        # Perform row-normalization
        for j in range(k_neighbors):
            P[i, j] /= sum_Pi

    return P


cpdef tuple estimate_positive_gradient_nn(
    sparse_index_type[:] indices,
    sparse_index_type[:] indptr,
    double[:] P_data,
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    double dof=1,
    Py_ssize_t num_threads=1,
    bint should_eval_error=False,
):
    cdef:
        Py_ssize_t n_samples = gradient.shape[0]
        Py_ssize_t n_dims = gradient.shape[1] 
        double * diff
        double d_ij, p_ij, q_ij, kl_divergence = 0, sum_P = 0
        double alpha_grad_pos = 0.0  # Initialize the positive alpha gradient term
        double s, log_s, frac       # Variables for alpha gradient calculation

        Py_ssize_t i, j, k, d

    if num_threads < 1:
        num_threads = 1

    # Degrees of freedom cannot be negative
    if dof <= 0:
        dof = 1e-8
    
    with nogil, parallel(num_threads=num_threads):
        # Use `malloc` here instead of `PyMem_Malloc` because we're in a
        # `nogil` clause and we won't be allocating much memory
        diff = <double *>malloc(n_dims * sizeof(double))
        if not diff:
            with gil:
                raise MemoryError()

        for i in prange(n_samples, schedule="guided"):
            # Iterate over all the neighbors `j` and sum up their contribution
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                p_ij = P_data[k]
                # Compute the direction of the points attraction and the
                # squared euclidean distance between the points
                d_ij = 0
                for d in range(n_dims):
                    diff[d] = embedding[i, d] - reference_embedding[j, d]
                    d_ij = d_ij + diff[d] * diff[d]

                if dof != 1:
                    # No need exp by dof here because the terms cancel out
                    q_ij = 1 / (1 + d_ij / dof)
                else:
                    q_ij = 1 / (1 + d_ij)

                # Compute F_{attr} of point `j` on point `i`
                for d in range(n_dims):
                    gradient[i, d] = gradient[i, d] + q_ij * p_ij * diff[d]

                # Compute the alpha gradient positive term
                s = 1.0 + d_ij / dof
                log_s = log(s)
                frac = d_ij / (dof + d_ij)
                alpha_grad_pos += p_ij * (log_s - frac)


                # Evaluating the following expressions can slow things down
                # considerably if evaluated every iteration. Note that the q_ij
                # is unnormalized, so we need to normalize once the sum of q_ij
                # is known
                if should_eval_error:
                    sum_P += p_ij
                    if dof != 1:
                        # Now we need to do the exp by dof
                        kl_divergence += p_ij * log((p_ij / (q_ij ** dof + EPSILON)) + EPSILON)                    
                    else:
                        kl_divergence += p_ij * log((p_ij / (q_ij + EPSILON)) + EPSILON)
        free(diff)

    return sum_P, kl_divergence, alpha_grad_pos


cpdef tuple estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=0.5,
    double dof=1,
    Py_ssize_t num_threads=1,
    bint pairwise_normalization=True,
):
    """Estimate the negative t-SNE gradient using the Barnes-Hut approximation.

    Returns
    -------
    sum_Q : double
        The sum of all q_{ij} values.
    alpha_grad_neg : double
        The negative term of the gradient with respect to alpha.
    
    Notes
    -----
    Changes the gradient inplace to avoid needless memory allocation. As
    such, this must be run before estimating the positive gradients, since
    the negative gradient must be normalized at the end with the sum of
    q_{ij}s.
    """
    cdef:
        Py_ssize_t i, num_points = embedding.shape[0]
        double sum_Q = 0.0
        double alpha_grad_neg = 0.0
        double[::1] sum_Qi = np.zeros(num_points, dtype=float)
        double[::1] alpha_grad_neg_i = np.zeros(num_points, dtype=float)

    if num_threads < 1:
        num_threads = 1

    # In order to run gradient estimation in parallel, we need to pass each
    # worker its own memory slot to write sum_Qs
    # Also estimate the negative part of the alpha-gradient   
    for i in prange(num_points,nogil=True, num_threads=num_threads, schedule="guided"):
        _estimate_negative_gradient_single(
            &tree.root,
            &embedding[i, 0],
            &gradient[i, 0],
            &sum_Qi[i],
            theta,
            dof,
            &alpha_grad_neg_i[i],
            embedding.shape[1]
        )

    # Aggregate sum_Q and alpha_grad_neg from all points
    for i in range(num_points):
        sum_Q += sum_Qi[i]
        alpha_grad_neg += alpha_grad_neg_i[i]

    # Normalize q_{ij}s
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if pairwise_normalization:
                gradient[i, j] /= sum_Q + EPSILON
            else:
                gradient[i, j] /= sum_Qi[i] + EPSILON

    alpha_grad_neg /= sum_Q # Normalize alpha_grad_neg
    return sum_Q, alpha_grad_neg


cdef void _estimate_negative_gradient_single(
    Node *node,
    double *point,
    double *gradient,
    double *sum_Q,
    double theta,
    double dof,
    double *alpha_grad_neg,
    Py_ssize_t n_dims
) noexcept nogil:
    cdef:
        double distance = EPSILON
        double q_ij, qij_term, tmp
        double grad_coeff  # Declare grad_coeff
        double s, log_s, frac
        Py_ssize_t d, i

    # Compute the squared euclidean distance in the embedding space from the
    # new point to the center of mass    
    for d in range(node.n_dims):
        tmp = node.center_of_mass[d] - point[d]
        distance += tmp * tmp

    # Degrees of freedom cannot be negative
    if dof <= 0:
        dof = 1e-8

    # Check if the node can be used as a summary (Barnes-Hut criterion)
    if node.is_leaf or node.length / sqrt(distance) < theta:
        if dof != 1:
            q_ij = 1 / pow(1.0 + distance/dof, dof)
        else:
            q_ij = 1 / (1 + distance)
        
        qij_term = q_ij*node.num_points  # q_ij * number of points in the node

        sum_Q[0] += qij_term

        # Compute the gradient contribution
        if dof != 1:
            grad_coeff = qij_term * pow(q_ij, 1.0 / dof)
        else:
            grad_coeff = qij_term * q_ij

        for d in range(n_dims):
            gradient[d] -= grad_coeff * (point[d] - node.center_of_mass[d])
        
        
        # Compute the negative alpha gradient contribution
        s = 1.0 + (distance / dof)
        log_s = log(s)
        frac = distance / (dof + distance)
        alpha_grad_neg[0] += qij_term * (log_s - frac)

        return

    # Recurse into child nodes
    for d in range(1 << node.n_dims):
        _estimate_negative_gradient_single(
            &node.children[d],
            point,
            gradient,
            sum_Q,
            theta,
            dof,
            alpha_grad_neg,
            n_dims
        )


# The kernel matrix is rebuilt every iteration because the grid follows the
# embedding's extent, and at 45k points it is a 1134x1134 array. Reusing the
# allocation avoids both the allocation and the page faults that come with
# touching fresh memory. One buffer per kernel, since when dof != 1 the plain
# and squared kernels are both live at once. Each thread keeps its own: the
# transforms release the GIL while they read a kernel, so embeddings optimized
# concurrently in different threads must not share one.
cdef object _kernel_buffers_local = threading.local()


cdef dict _kernel_buffers():
    cdef object buffers = getattr(_kernel_buffers_local, "buffers", None)
    if buffers is None:
        buffers = _kernel_buffers_local.buffers = {}
    return buffers


cdef double[:, ::1] _kernel_buffer_2d(Py_ssize_t size, bint exp1p):
    cdef dict buffers = _kernel_buffers()
    cdef object buf = buffers.get(exp1p)
    if buf is None or buf.shape[0] != size:
        buf = np.zeros((size, size), dtype=float)
        buffers[exp1p] = buf
    return buf


cdef double[::1] _kernel_buffer_1d(Py_ssize_t size, bint exp1p):
    cdef dict buffers = _kernel_buffers()
    cdef object buf = buffers.get(("1d", exp1p))
    if buf is None or buf.shape[0] != size:
        buf = np.zeros(size, dtype=float)
        buffers[("1d", exp1p)] = buf
    return buf


cdef double[:, ::1] interpolate(double[::1] y_in_box, double[::1] y_tilde):
    """Lagrangian polynomial interpolation."""
    cdef Py_ssize_t N = y_in_box.shape[0]
    cdef Py_ssize_t n_interpolation_points = y_tilde.shape[0]

    cdef double[:, ::1] interpolated_values = np.empty((N, n_interpolation_points), dtype=float)
    cdef double[::1] denominator = np.empty(n_interpolation_points, dtype=float)
    cdef Py_ssize_t i, j, k

    for i in range(n_interpolation_points):
        denominator[i] = 1
        for j in range(n_interpolation_points):
            if i != j:
                denominator[i] *= y_tilde[i] - y_tilde[j]

    for i in range(N):
        for j in range(n_interpolation_points):
            interpolated_values[i, j] = 1
            for k in range(n_interpolation_points):
                if j != k:
                    interpolated_values[i, j] *= y_in_box[i] - y_tilde[k]
            interpolated_values[i, j] /= denominator[j]

    return interpolated_values


cdef double[::1] compute_kernel_tilde_1d(
    bint exp1p,
    Py_ssize_t n_interpolation_points_1d,
    double coord_min,
    double coord_spacing,
    double dof,
):
    cdef:
        double[::1] y_tilde = np.empty(n_interpolation_points_1d, dtype=float)

        Py_ssize_t embedded_size = 2 * n_interpolation_points_1d
        double[::1] kernel_tilde = _kernel_buffer_1d(embedded_size, exp1p)

        Py_ssize_t i
        double y_diff, tmp, exponent
        bint dof_is_one = dof == 1

    y_tilde[0] = coord_spacing / 2 + coord_min
    for i in range(1, n_interpolation_points_1d):
        y_tilde[i] = y_tilde[i - 1] + coord_spacing

    if exp1p:
        exponent = -(dof + 1)
    else:
        exponent = -dof

    # Evaluate the kernel at the interpolation nodes and form the embedded
    # generating kernel vector for a circulant matrix. Every entry but the
    # first is written, so a reused buffer needs no clearing
    for i in range(n_interpolation_points_1d):
        y_diff = y_tilde[0] - y_tilde[i]

        if dof_is_one:
            tmp = 1 + y_diff * y_diff
            if exp1p:
                tmp = 1 / (tmp * tmp)
            else:
                tmp = 1 / tmp
        else:
            tmp = (1 + (y_diff * y_diff) / dof) ** exponent

        kernel_tilde[n_interpolation_points_1d + i] = tmp
        kernel_tilde[n_interpolation_points_1d - i] = tmp

    return kernel_tilde


cpdef double estimate_negative_gradient_fft_1d(
    double[::1] embedding,
    double[::1] gradient,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    Py_ssize_t num_threads=1,
):
    cdef Py_ssize_t i, j, d, box_idx, n_samples = embedding.shape[0]
    cdef double y_max = -INFINITY, y_min = INFINITY
    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    for i in range(n_samples):
        if embedding[i] < y_min:
            y_min = embedding[i]
        elif embedding[i] > y_max:
            y_max = embedding[i]

    cdef int n_boxes = <int>fmax(min_num_intervals, (y_max - y_min) / ints_in_interval)
    # FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    # 11^e 13^f, where e+f is either 0 or 1, and the other exponents are arbitrary
    cdef list recommended_boxes = [
        20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50,
        52, 54, 55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
        98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 125, 126, 128, 130, 132, 135, 140,
        144, 147, 150, 154, 156, 160, 162, 165, 168, 175, 176, 180, 182, 189, 192, 195, 196,
        198, 200, 208, 210, 216, 220, 224, 225, 231, 234, 240, 243, 245, 250, 252, 256, 260,
        264, 270, 273, 275, 280, 288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336,
        343, 350, 351, 352, 360, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416, 420,
        432, 440, 441, 448, 450, 455, 462, 468, 480, 486, 490, 495, 500, 504, 512, 520, 525,
        528, 539, 540, 546, 550, 560, 567, 576, 585, 588, 594, 600, 616, 624, 625, 630, 637,
        640, 648, 650, 660, 672, 675, 686, 693, 700, 702, 704, 720, 728, 729, 735, 750, 756,
        768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 864, 875, 880, 882, 891, 896,
        900, 910, 924, 936, 945, 960, 972, 975, 980, 990, 1000 
    ]
    if n_boxes < recommended_boxes[205]:
        i = 0
        while n_boxes > recommended_boxes[i]:
            i += 1
        n_boxes = recommended_boxes[i]
    else:
        n_boxes = 1000

    cdef double box_width = (y_max - y_min) / n_boxes

    # Compute the box bounds
    cdef double[::1] box_lower_bounds = np.empty(n_boxes, dtype=float)
    cdef double[::1] box_upper_bounds = np.empty(n_boxes, dtype=float)
    for box_idx in range(n_boxes):
        box_lower_bounds[box_idx] = box_idx * box_width + y_min
        box_upper_bounds[box_idx] = (box_idx + 1) * box_width + y_min

    # Determine which box each point belongs to
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    for i in range(n_samples):
        box_idx = <int>((embedding[i] - y_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_idx >= n_boxes:
            box_idx = n_boxes - 1

        point_box_idx[i] = box_idx

    cdef int n_interpolation_points_1d = n_interpolation_points * n_boxes
    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the the squared cauchy kernel at the interpolation nodes
    cdef double[::1] sq_kernel_tilde = compute_kernel_tilde_1d(
        True, n_interpolation_points_1d, y_min, h * box_width, dof
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef double[::1] kernel_tilde
    if dof != 1:
        kernel_tilde = compute_kernel_tilde_1d(
            False, n_interpolation_points_1d, y_min, h * box_width, dof
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    cdef int n_terms = 3
    cdef double[:, ::1] q_j = np.empty((n_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i]
            q_j[i, 2] = 1
    else:
        for i in range(n_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i]
            q_j[i, 2] = embedding[i] ** 2

    # Compute the relative position of each reference point in its box
    cdef double[::1] y_in_box = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        y_in_box[i] = (embedding[i] - box_lower_bounds[box_idx]) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] interpolated_values = interpolate(y_in_box, y_tilde)

    # Actually compute w_{ij}s
    cdef double[:, ::1] w_coefficients = np.zeros((n_interpolation_points_1d, n_terms), dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                w_coefficients[box_idx + j, d] += interpolated_values[i, j] * q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef double[:, ::1] y_tilde_values = np.empty((n_interpolation_points_1d, n_terms))
    if dof != 1:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients[:, :2], y_tilde_values[:, :2], num_threads)
        matrix_multiply_fft_1d(kernel_tilde, w_coefficients[:, 2:], y_tilde_values[:, 2:], num_threads)
    else:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)


    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    cdef double[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                phi[i, d] += interpolated_values[i, j] * y_tilde_values[box_idx + j, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double sum_Q = 0
    if dof != 1:
        for i in range(n_samples):
            sum_Q += phi[i, 2]
    else:
        for i in range(n_samples):
            sum_Q += (1 + embedding[i] ** 2) * phi[i, 0] - \
                     2 * embedding[i] * phi[i, 1] + \
                     phi[i, 2]

    sum_Q -= n_samples

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i] -= (embedding[i] * phi[i, 0] - phi[i, 1]) / (sum_Q + EPSILON)

    return sum_Q


cpdef tuple prepare_negative_gradient_fft_interpolation_grid_1d(
    double[::1] reference_embedding,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    double padding=0,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_reference_samples = reference_embedding.shape[0]

        double y_max = -INFINITY, y_min = INFINITY
    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    # First, check the existing embedding
    for i in range(n_reference_samples):
        if reference_embedding[i] < y_min:
            y_min = reference_embedding[i]
        elif reference_embedding[i] > y_max:
            y_max = reference_embedding[i]

    # We assume here that the embedding is centered and we want to generate an
    # equal grid in both negative and positive lines
    if fabs(y_min) > fabs(y_max):
        coord_max = -y_min
    elif fabs(y_max) > fabs(y_min):
        coord_min = -y_max

    # Apply padding to the min/max coordinates
    y_min *= 1 + padding
    y_max *= 1 + padding

    cdef int n_boxes = <int>fmax(min_num_intervals, (y_max - y_min) / ints_in_interval)
    cdef double box_width = (y_max - y_min) / n_boxes

    # Compute the box bounds
    cdef double[::1] box_lower_bounds = np.empty(n_boxes, dtype=float)
    cdef double[::1] box_upper_bounds = np.empty(n_boxes, dtype=float)
    for box_idx in range(n_boxes):
        box_lower_bounds[box_idx] = box_idx * box_width + y_min
        box_upper_bounds[box_idx] = (box_idx + 1) * box_width + y_min

    # Determine which box each reference point belongs to
    cdef int *reference_point_box_idx = <int *>PyMem_Malloc(n_reference_samples * sizeof(int))
    for i in range(n_reference_samples):
        box_idx = <int>((reference_embedding[i] - y_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_idx >= n_boxes:
            box_idx = n_boxes - 1

        reference_point_box_idx[i] = box_idx

    cdef int n_interpolation_points_1d = n_interpolation_points * n_boxes
    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the squared cauchy kernel at the interpolation nodes
    cdef double[::1] sq_kernel_tilde = compute_kernel_tilde_1d(
        True, n_interpolation_points_1d, y_min, h * box_width, dof
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef double[::1] kernel_tilde
    if dof != 1:
        kernel_tilde = compute_kernel_tilde_1d(
            False, n_interpolation_points_1d, y_min, h * box_width, dof
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    cdef int n_terms = 3
    cdef double[:, ::1] q_j = np.empty((n_reference_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i]
            q_j[i, 2] = 1
    else:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i]
            q_j[i, 2] = reference_embedding[i] ** 2

    # Compute the relative position of each reference point in its box
    cdef double[::1] reference_y_in_box = np.empty(n_reference_samples, dtype=float)
    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i]
        reference_y_in_box[i] = (reference_embedding[i] - box_lower_bounds[box_idx]) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] reference_interpolated_values = interpolate(reference_y_in_box, y_tilde)

    # Actually compute w_{ij}s
    cdef double[:, ::1] w_coefficients = np.zeros((n_interpolation_points_1d, n_terms), dtype=float)
    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                w_coefficients[box_idx + j, d] += reference_interpolated_values[i, j] * q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef double[:, ::1] y_tilde_values = np.empty((n_interpolation_points_1d, n_terms))
    if dof != 1:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients[:, :2], y_tilde_values[:, :2], num_threads)
        matrix_multiply_fft_1d(kernel_tilde, w_coefficients[:, 2:], y_tilde_values[:, 2:], num_threads)
    else:
        matrix_multiply_fft_1d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    PyMem_Free(reference_point_box_idx)

    return np.asarray(y_tilde_values), np.asarray(box_lower_bounds)


cpdef double estimate_negative_gradient_fft_1d_with_grid(
    double[::1] embedding,
    double[::1] gradient,
    double[:, ::1] y_tilde_values,
    double[::1] box_lower_bounds,
    Py_ssize_t n_interpolation_points,
    double dof,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_terms = y_tilde_values.shape[1]
        Py_ssize_t n_boxes = box_lower_bounds.shape[0]
        double y_min = box_lower_bounds[0]
        double box_width = box_lower_bounds[1] - box_lower_bounds[0]

    # Determine which box each point belongs to
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    for i in range(n_samples):
        box_idx = <int>((embedding[i] - y_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_idx >= n_boxes:
            box_idx = n_boxes - 1

        point_box_idx[i] = box_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    # Compute the relative position of each new embedding point in its box
    cdef double[::1] y_in_box = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        y_in_box[i] = (embedding[i] - box_lower_bounds[box_idx]) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] interpolated_values = interpolate(y_in_box, y_tilde)

    # Actually compute \tilde{\phi(y_i)}
    cdef double[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i] * n_interpolation_points
        for j in range(n_interpolation_points):
            for d in range(n_terms):
                phi[i, d] += interpolated_values[i, j] * y_tilde_values[box_idx + j, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double[::1] sum_Qi = np.empty(n_samples, dtype=float)
    if dof != 1:
        for i in range(n_samples):
            sum_Qi[i] = phi[i, 2]
    else:
        for i in range(n_samples):
            sum_Qi[i] = (1 + embedding[i] ** 2) * phi[i, 0] - \
                         2 * embedding[i] * phi[i, 1] + \
                         phi[i, 2]

    cdef double sum_Q = 0
    for i in range(n_samples):
        sum_Q += sum_Qi[i]

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i] -= (embedding[i] * phi[i, 0] - phi[i, 1]) / (sum_Qi[i] + EPSILON)

    return sum_Q


cdef double[:, ::1] compute_kernel_tilde_2d(
    bint exp1p,
    Py_ssize_t n_interpolation_points_1d,
    double coord_min,
    double coord_spacing,
    double dof,
):
    """Evaluate the kernel at every pair of interpolation nodes.

    `exp1p` selects the squared Cauchy kernel over the plain one. It is a flag
    rather than a function pointer because this is a quarter of a million
    evaluations per call and an indirect call cannot be inlined; with the
    kernel written out here, the `dof` branch is hoisted out of the loop and
    the whole body collapses to a handful of instructions.
    """
    cdef:
        double[::1] y_tilde = np.empty(n_interpolation_points_1d, dtype=float)
        double[::1] x_tilde = np.empty(n_interpolation_points_1d, dtype=float)

        Py_ssize_t embedded_size = 2 * n_interpolation_points_1d
        double[:, ::1] kernel_tilde = _kernel_buffer_2d(embedded_size, exp1p)

        Py_ssize_t i, j
        double y_diff, x_diff, tmp, exponent
        bint dof_is_one = dof == 1
        double * row_above
        double * row_below

    x_tilde[0] = coord_min + coord_spacing / 2
    y_tilde[0] = coord_min + coord_spacing / 2
    for i in range(1, n_interpolation_points_1d):
        x_tilde[i] = x_tilde[i - 1] + coord_spacing
        y_tilde[i] = y_tilde[i - 1] + coord_spacing

    if exp1p:
        exponent = -(dof + 1)
    else:
        exponent = -dof

    # Evaluate the kernel at the interpolation nodes and form the embedded
    # generating kernel matrix for a circulant matrix. Everything outside the
    # first row and column is written, so a reused buffer needs no clearing
    for i in range(n_interpolation_points_1d):
        y_diff = y_tilde[0] - y_tilde[i]
        row_above = &kernel_tilde[n_interpolation_points_1d + i, 0]
        row_below = &kernel_tilde[n_interpolation_points_1d - i, 0]

        for j in range(n_interpolation_points_1d):
            x_diff = x_tilde[0] - x_tilde[j]

            if dof_is_one:
                tmp = 1 + y_diff * y_diff + x_diff * x_diff
                if exp1p:
                    tmp = 1 / (tmp * tmp)
                else:
                    tmp = 1 / tmp
            else:
                tmp = (1 + (y_diff * y_diff + x_diff * x_diff) / dof) ** exponent

            row_above[n_interpolation_points_1d + j] = tmp
            row_below[n_interpolation_points_1d + j] = tmp
            row_above[n_interpolation_points_1d - j] = tmp
            row_below[n_interpolation_points_1d - j] = tmp

    return kernel_tilde


cpdef double estimate_negative_gradient_fft_2d(
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_dims = embedding.shape[1]

        double coord_max = -INFINITY, coord_min = INFINITY
    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    for i in range(n_samples):
        if embedding[i, 0] < coord_min:
            coord_min = embedding[i, 0]
        elif embedding[i, 0] > coord_max:
            coord_max = embedding[i, 0]
        if embedding[i, 1] < coord_min:
            coord_min = embedding[i, 1]
        elif embedding[i, 1] > coord_max:
            coord_max = embedding[i, 1]

    cdef int n_boxes_1d = <int>fmax(min_num_intervals, (coord_max - coord_min) / ints_in_interval)
    # FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    # 11^e 13^f, where e+f is either 0 or 1, and the other exponents are arbitrary
    cdef list recommended_boxes = [
        20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 39, 40, 42, 44, 45, 48, 49, 50,
        52, 54, 55, 56, 60, 63, 64, 65, 66, 70, 72, 75, 77, 78, 80, 81, 84, 88, 90, 91, 96,
        98, 99, 100, 104, 105, 108, 110, 112, 117, 120, 125, 126, 128, 130, 132, 135, 140,
        144, 147, 150, 154, 156, 160, 162, 165, 168, 175, 176, 180, 182, 189, 192, 195, 196,
        198, 200, 208, 210, 216, 220, 224, 225, 231, 234, 240, 243, 245, 250, 252, 256, 260,
        264, 270, 273, 275, 280, 288, 294, 297, 300, 308, 312, 315, 320, 324, 325, 330, 336,
        343, 350, 351, 352, 360, 364, 375, 378, 384, 385, 390, 392, 396, 400, 405, 416, 420,
        432, 440, 441, 448, 450, 455, 462, 468, 480, 486, 490, 495, 500, 504, 512, 520, 525,
        528, 539, 540, 546, 550, 560, 567, 576, 585, 588, 594, 600, 616, 624, 625, 630, 637,
        640, 648, 650, 660, 672, 675, 686, 693, 700, 702, 704, 720, 728, 729, 735, 750, 756,
        768, 770, 780, 784, 792, 800, 810, 819, 825, 832, 840, 864, 875, 880, 882, 891, 896,
        900, 910, 924, 936, 945, 960, 972, 975, 980, 990, 1000 
    ]
    if n_boxes_1d < recommended_boxes[205]:
        i = 0
        while n_boxes_1d > recommended_boxes[i]:
            i += 1
        n_boxes_1d = recommended_boxes[i]
    else:
        n_boxes_1d = 1000 

    cdef int n_total_boxes = n_boxes_1d ** 2
    cdef double box_width = (coord_max - coord_min) / n_boxes_1d

    # Compute the box bounds
    cdef:
        double[::1] box_x_lower_bounds = np.empty(n_total_boxes, dtype=float)
        double[::1] box_x_upper_bounds = np.empty(n_total_boxes, dtype=float)
        double[::1] box_y_lower_bounds = np.empty(n_total_boxes, dtype=float)
        double[::1] box_y_upper_bounds = np.empty(n_total_boxes, dtype=float)

    for i in range(n_boxes_1d):
        for j in range(n_boxes_1d):
            box_x_lower_bounds[i * n_boxes_1d + j] = j * box_width + coord_min
            box_x_upper_bounds[i * n_boxes_1d + j] = (j + 1) * box_width + coord_min

            box_y_lower_bounds[i * n_boxes_1d + j] = i * box_width + coord_min
            box_y_upper_bounds[i * n_boxes_1d + j] = (i + 1) * box_width + coord_min

    # Determine which box each reference point belongs to
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    cdef int box_x_idx, box_y_idx
    for i in range(n_samples):
        box_x_idx = <int>((embedding[i, 0] - coord_min) / box_width)
        box_y_idx = <int>((embedding[i, 1] - coord_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_x_idx >= n_boxes_1d:
            box_x_idx = n_boxes_1d - 1
        if box_y_idx >= n_boxes_1d:
            box_y_idx = n_boxes_1d - 1

        point_box_idx[i] = box_y_idx * n_boxes_1d + box_x_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the squared cauchy kernel at the interpolation nodes
    cdef double[:, ::1] sq_kernel_tilde = compute_kernel_tilde_2d(
         True, n_interpolation_points * n_boxes_1d, coord_min, h * box_width, dof,
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef double[:, ::1] kernel_tilde
    if dof != 1:
        kernel_tilde = compute_kernel_tilde_2d(
            False, n_interpolation_points * n_boxes_1d, coord_min, h * box_width, dof,
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    cdef int n_terms = 4
    cdef double[:, ::1] q_j = np.empty((n_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i, 0]
            q_j[i, 2] = embedding[i, 1]
            q_j[i, 3] = 1
    else:
        for i in range(n_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = embedding[i, 0]
            q_j[i, 2] = embedding[i, 1]
            q_j[i, 3] = embedding[i, 0] ** 2 + embedding[i, 1] ** 2

    # Compute the relative position of each reference point in its box
    cdef:
        double[::1] x_in_box = np.empty(n_samples, dtype=float)
        double[::1] y_in_box = np.empty(n_samples, dtype=float)
        double y_min, x_min

    for i in range(n_samples):
        box_idx = point_box_idx[i]
        x_min = box_x_lower_bounds[box_idx]
        y_min = box_y_lower_bounds[box_idx]
        x_in_box[i] = (embedding[i, 0] - x_min) / box_width
        y_in_box[i] = (embedding[i, 1] - y_min) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] x_interpolated_values = interpolate(x_in_box, y_tilde)
    cdef double[:, ::1] y_interpolated_values = interpolate(y_in_box, y_tilde)

    # Actually compute w_{ij}s
    cdef:
        int total_interpolation_points = n_total_boxes * n_interpolation_points ** 2
        double[:, ::1] w_coefficients = np.zeros((total_interpolation_points, n_terms), dtype=float)
        Py_ssize_t box_i, box_j, interp_i, interp_j, idx

    for i in range(n_samples):
        box_idx = point_box_idx[i]
        box_i = box_idx % n_boxes_1d
        box_j = box_idx // n_boxes_1d
        for interp_i in range(n_interpolation_points):
            for interp_j in range(n_interpolation_points):
                idx = (box_i * n_interpolation_points + interp_i) * \
                      (n_boxes_1d * n_interpolation_points) + \
                      (box_j * n_interpolation_points) + \
                      interp_j
                for d in range(n_terms):
                    w_coefficients[idx, d] += \
                        x_interpolated_values[i, interp_i] * \
                        y_interpolated_values[i, interp_j] * \
                        q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef double[:, ::1] y_tilde_values = np.empty((total_interpolation_points, n_terms))
    if dof != 1:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients[:, :3], y_tilde_values[:, :3], num_threads)
        matrix_multiply_fft_2d(kernel_tilde, w_coefficients[:, 3:], y_tilde_values[:, 3:], num_threads)
    else:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    cdef double[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        box_i = box_idx % n_boxes_1d
        box_j = box_idx // n_boxes_1d
        for interp_i in range(n_interpolation_points):
            for interp_j in range(n_interpolation_points):
                idx = (box_i * n_interpolation_points + interp_i) * \
                      (n_boxes_1d * n_interpolation_points) + \
                      (box_j * n_interpolation_points) + \
                      interp_j
                for d in range(n_terms):
                    phi[i, d] += x_interpolated_values[i, interp_i] * \
                                 y_interpolated_values[i, interp_j] * \
                                 y_tilde_values[idx, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double sum_Q = 0, y1, y2
    if dof != 1:
        for i in range(n_samples):
            sum_Q += phi[i, 3]
    else:
        for i in range(n_samples):
            y1 = embedding[i, 0]
            y2 = embedding[i, 1]

            sum_Q += (1 + y1 ** 2 + y2 ** 2) * phi[i, 0] - \
                     2 * (y1 * phi[i, 1] + y2 * phi[i, 2]) + \
                     phi[i, 3]

    sum_Q -= n_samples

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i, 0] -= (embedding[i, 0] * phi[i, 0] - phi[i, 1]) / (sum_Q + EPSILON)
        gradient[i, 1] -= (embedding[i, 1] * phi[i, 0] - phi[i, 2]) / (sum_Q + EPSILON)

    return sum_Q


cpdef tuple prepare_negative_gradient_fft_interpolation_grid_2d(
    double[:, ::1] reference_embedding,
    Py_ssize_t n_interpolation_points=3,
    Py_ssize_t min_num_intervals=10,
    double ints_in_interval=1,
    double dof=1,
    double padding=0,
    Py_ssize_t num_threads=1,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_reference_samples = reference_embedding.shape[0]

        double coord_max = -INFINITY, coord_min = INFINITY
    if num_threads < 1:
        num_threads = 1

    # Determine the min/max values of the embedding
    # First, check the existing embedding
    for i in range(n_reference_samples):
        if reference_embedding[i, 0] < coord_min:
            coord_min = reference_embedding[i, 0]
        elif reference_embedding[i, 0] > coord_max:
            coord_max = reference_embedding[i, 0]
        if reference_embedding[i, 1] < coord_min:
            coord_min = reference_embedding[i, 1]
        elif reference_embedding[i, 1] > coord_max:
            coord_max = reference_embedding[i, 1]

    # We assume here that the embedding is centered and we want to generate an
    # equal grid in all quadrants
    if fabs(coord_min) > fabs(coord_max):
        coord_max = -coord_min
    elif fabs(coord_max) > fabs(coord_min):
        coord_min = -coord_max

    # Apply padding to the min/max coordinates
    coord_min *= 1 + padding
    coord_max *= 1 + padding

    cdef int n_boxes_1d = <int>fmax(min_num_intervals, (coord_max - coord_min) / ints_in_interval)
    cdef int n_total_boxes = n_boxes_1d ** 2
    cdef double box_width = (coord_max - coord_min) / n_boxes_1d

    # Compute the box bounds
    cdef:
        double[::1] box_x_lower_bounds = np.empty(n_total_boxes, dtype=float)
        double[::1] box_x_upper_bounds = np.empty(n_total_boxes, dtype=float)
        double[::1] box_y_lower_bounds = np.empty(n_total_boxes, dtype=float)
        double[::1] box_y_upper_bounds = np.empty(n_total_boxes, dtype=float)

    for i in range(n_boxes_1d):
        for j in range(n_boxes_1d):
            box_x_lower_bounds[i * n_boxes_1d + j] = j * box_width + coord_min
            box_x_upper_bounds[i * n_boxes_1d + j] = (j + 1) * box_width + coord_min

            box_y_lower_bounds[i * n_boxes_1d + j] = i * box_width + coord_min
            box_y_upper_bounds[i * n_boxes_1d + j] = (i + 1) * box_width + coord_min

    # Determine which box each reference point belongs to
    cdef int *reference_point_box_idx = <int *>PyMem_Malloc(n_reference_samples * sizeof(int))
    cdef int box_x_idx, box_y_idx
    for i in range(n_reference_samples):
        box_x_idx = <int>((reference_embedding[i, 0] - coord_min) / box_width)
        box_y_idx = <int>((reference_embedding[i, 1] - coord_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_x_idx >= n_boxes_1d:
            box_x_idx = n_boxes_1d - 1
        if box_y_idx >= n_boxes_1d:
            box_y_idx = n_boxes_1d - 1

        reference_point_box_idx[i] = box_y_idx * n_boxes_1d + box_x_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # Evaluate the the squared cauchy kernel at the interpolation nodes
    cdef double[:, ::1] sq_kernel_tilde = compute_kernel_tilde_2d(
         True, n_interpolation_points * n_boxes_1d, coord_min, h * box_width, dof,
    )
    # The non-square cauchy kernel is only used if dof != 1, so don't do unnecessary work
    cdef double[:, ::1] kernel_tilde
    if dof != 1:
        kernel_tilde = compute_kernel_tilde_2d(
            False, n_interpolation_points * n_boxes_1d, coord_min, h * box_width, dof,
        )

    # STEP 1: Compute the w coefficients
    # Set up q_j values
    cdef int n_terms = 4
    cdef double[:, ::1] q_j = np.empty((n_reference_samples, n_terms), dtype=float)
    if dof != 1:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i, 0]
            q_j[i, 2] = reference_embedding[i, 1]
            q_j[i, 3] = 1
    else:
        for i in range(n_reference_samples):
            q_j[i, 0] = 1
            q_j[i, 1] = reference_embedding[i, 0]
            q_j[i, 2] = reference_embedding[i, 1]
            q_j[i, 3] = reference_embedding[i, 0] ** 2 + reference_embedding[i, 1] ** 2

    # Compute the relative position of each reference point in its box
    cdef:
        double[::1] reference_x_in_box = np.empty(n_reference_samples, dtype=float)
        double[::1] reference_y_in_box = np.empty(n_reference_samples, dtype=float)
        double y_min, x_min

    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i]
        x_min = box_x_lower_bounds[box_idx]
        y_min = box_y_lower_bounds[box_idx]
        reference_x_in_box[i] = (reference_embedding[i, 0] - x_min) / box_width
        reference_y_in_box[i] = (reference_embedding[i, 1] - y_min) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] reference_x_interpolated_values = interpolate(reference_x_in_box, y_tilde)
    cdef double[:, ::1] reference_y_interpolated_values = interpolate(reference_y_in_box, y_tilde)

    # Actually compute w_{ij}s
    cdef:
        int total_interpolation_points = n_total_boxes * n_interpolation_points ** 2
        double[:, ::1] w_coefficients = np.zeros((total_interpolation_points, n_terms), dtype=float)
        Py_ssize_t box_i, box_j, interp_i, interp_j, idx

    for i in range(n_reference_samples):
        box_idx = reference_point_box_idx[i]
        box_i = box_idx % n_boxes_1d
        box_j = box_idx // n_boxes_1d
        for interp_i in range(n_interpolation_points):
            for interp_j in range(n_interpolation_points):
                idx = (box_i * n_interpolation_points + interp_i) * \
                      (n_boxes_1d * n_interpolation_points) + \
                      (box_j * n_interpolation_points) + \
                      interp_j
                for d in range(n_terms):
                    w_coefficients[idx, d] += \
                        reference_x_interpolated_values[i, interp_i] * \
                        reference_y_interpolated_values[i, interp_j] * \
                        q_j[i, d]

    # STEP 2: Compute the kernel values evaluated at the interpolation nodes
    cdef double[:, ::1] y_tilde_values = np.empty((total_interpolation_points, n_terms))
    if dof != 1:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients[:, :3], y_tilde_values[:, :3], num_threads)
        matrix_multiply_fft_2d(kernel_tilde, w_coefficients[:, 3:], y_tilde_values[:, 3:], num_threads)
    else:
        matrix_multiply_fft_2d(sq_kernel_tilde, w_coefficients, y_tilde_values, num_threads)

    return (
        np.asarray(y_tilde_values),
        np.asarray(box_x_lower_bounds),
        np.asarray(box_y_lower_bounds),
    )


cpdef double estimate_negative_gradient_fft_2d_with_grid(
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double[:, ::1] y_tilde_values,
    double[::1] box_x_lower_bounds,
    double[::1] box_y_lower_bounds,
    Py_ssize_t n_interpolation_points,
    double dof,
):
    cdef:
        Py_ssize_t i, j, d, box_idx
        Py_ssize_t n_samples = embedding.shape[0]
        Py_ssize_t n_terms = y_tilde_values.shape[1]
        Py_ssize_t n_boxes_1d = int(sqrt(box_x_lower_bounds.shape[0]))
        double coord_min = box_x_lower_bounds[0]
        double box_width = box_x_lower_bounds[1] - box_x_lower_bounds[0]

    # Determine which box each point belongs to
    cdef int box_x_idx, box_y_idx
    cdef int *point_box_idx = <int *>PyMem_Malloc(n_samples * sizeof(int))
    for i in range(n_samples):
        box_x_idx = <int>((embedding[i, 0] - coord_min) / box_width)
        box_y_idx = <int>((embedding[i, 1] - coord_min) / box_width)
        # The right most point maps directly into `n_boxes`, while it should
        # belong to the last box
        if box_x_idx >= n_boxes_1d:
            box_x_idx = n_boxes_1d - 1
        if box_y_idx >= n_boxes_1d:
            box_y_idx = n_boxes_1d - 1

        point_box_idx[i] = box_y_idx * n_boxes_1d + box_x_idx

    # Prepare the interpolants for a single interval, so we can use their
    # relative positions later on
    cdef double[::1] y_tilde = np.empty(n_interpolation_points, dtype=float)
    cdef double h = 1. / n_interpolation_points
    y_tilde[0] = h / 2
    for i in range(1, n_interpolation_points):
        y_tilde[i] = y_tilde[i - 1] + h

    # STEP 3: Compute the potentials \tilde{\phi(y_i)}
    # Compute the relative position of each new embedding point in its box
    cdef:
        double[::1] x_in_box = np.empty(n_samples, dtype=float)
        double[::1] y_in_box = np.empty(n_samples, dtype=float)

    cdef double y_min, x_min
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        x_min = box_x_lower_bounds[box_idx]
        y_min = box_y_lower_bounds[box_idx]
        x_in_box[i] = (embedding[i, 0] - x_min) / box_width
        y_in_box[i] = (embedding[i, 1] - y_min) / box_width

    # Interpolate kernel using Lagrange polynomials
    cdef double[:, ::1] x_interpolated_values = interpolate(x_in_box, y_tilde)
    cdef double[:, ::1] y_interpolated_values = interpolate(y_in_box, y_tilde)

    # Actually compute \tilde{\phi(y_i)}
    cdef Py_ssize_t box_i, box_j, interp_i, interp_j, idx

    cdef double[:, ::1] phi = np.zeros((n_samples, n_terms), dtype=float)
    for i in range(n_samples):
        box_idx = point_box_idx[i]
        box_i = box_idx % n_boxes_1d
        box_j = box_idx // n_boxes_1d
        for interp_i in range(n_interpolation_points):
            for interp_j in range(n_interpolation_points):
                idx = (box_i * n_interpolation_points + interp_i) * \
                      (n_boxes_1d * n_interpolation_points) + \
                      (box_j * n_interpolation_points) + \
                      interp_j
                for d in range(n_terms):
                    phi[i, d] += x_interpolated_values[i, interp_i] * \
                                 y_interpolated_values[i, interp_j] * \
                                 y_tilde_values[idx, d]

    PyMem_Free(point_box_idx)

    # Compute the normalization term Z or sum of q_{ij}s
    cdef double[::1] sum_Qi = np.empty(n_samples, dtype=float)
    cdef double y1, y2
    if dof != 1:
        for i in range(n_samples):
            sum_Qi[i] = phi[i, 3]
    else:
        for i in range(n_samples):
            y1 = embedding[i, 0]
            y2 = embedding[i, 1]

            sum_Qi[i] = (1 + y1 ** 2 + y2 ** 2) * phi[i, 0] - \
                        2 * (y1 * phi[i, 1] + y2 * phi[i, 2]) + \
                        phi[i, 3]

    cdef sum_Q = 0
    for i in range(n_samples):
        sum_Q += sum_Qi[i]

    # The phis used here are not affected if dof != 1
    for i in range(n_samples):
        gradient[i, 0] -= (embedding[i, 0] * phi[i, 0] - phi[i, 1]) / (sum_Qi[i] + EPSILON)
        gradient[i, 1] -= (embedding[i, 1] * phi[i, 0] - phi[i, 2]) / (sum_Qi[i] + EPSILON)

    return sum_Q
