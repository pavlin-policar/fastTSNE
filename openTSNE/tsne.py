import inspect
import logging
import multiprocessing
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from time import time
from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator

from openTSNE import _tsne, utils
from openTSNE import initialization as initialization_scheme
from openTSNE.affinity import Affinities, MultiscaleMixture
from openTSNE.quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


@dataclass
class GradientResult:
    """Result of a single objective-function evaluation."""

    error: float
    gradient: np.ndarray
    dof_grad: float = 0.0


@dataclass
class IterationState:
    """Per-iteration snapshot passed to optimization callbacks.

    Constructed only when a callback fires, and contains copies of the
    embedding and gradient so callbacks can safely retain the state.
    """

    iteration: int
    exaggeration: float
    embedding: np.ndarray
    gradient: np.ndarray
    error: float
    dof: float
    dof_grad: float


def _adapt_callback(c):
    """Wrap a user callback so the optimizer can always invoke it as `c(state)`.

    Old-style 3-arg callbacks `c(iteration, error, embedding)` are still
    accepted but emit a FutureWarning at registration time.
    """
    try:
        sig = inspect.signature(c)
    except (TypeError, ValueError):
        # Builtins or C-extensions without introspectable signatures: assume new style.
        return c

    required = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    n_required = len(required)
    accepts_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )

    if n_required == 1 or (n_required == 0 and accepts_var_positional):
        return c
    if n_required == 3:
        warnings.warn(
            "Callbacks taking (iteration, error, embedding) are deprecated and "
            "will be removed in a future release. Switch to a single-argument "
            "callback `callback(state: IterationState)` to access the full "
            "per-iteration state (including dof, gradient, exaggeration, etc.).",
            FutureWarning,
            stacklevel=3,
        )

        def wrapped(state, _c=c):
            return _c(state.iteration, state.error, state.embedding)

        # Preserve the optimization_about_to_start hook (used by class-based
        # callbacks subclassing `Callback`) so it still fires through the wrapper.
        if hasattr(c, "optimization_about_to_start"):
            wrapped.optimization_about_to_start = c.optimization_about_to_start
        return wrapped
    raise TypeError(
        "Callback must accept either a single IterationState argument or the "
        "deprecated (iteration, error, embedding) signature; got a callable "
        f"requiring {n_required} positional arguments."
    )


def _check_callbacks(callbacks):
    if callbacks is not None:
        # If list was passed, make sure all of them are actually callable
        if isinstance(callbacks, Iterable):
            if any(not callable(c) for c in callbacks):
                raise ValueError("`callbacks` must contain callable objects!")
        # The gradient descent method deals with lists
        elif callable(callbacks):
            callbacks = (callbacks,)
        else:
            raise ValueError("`callbacks` must be a callable object!")

        callbacks = tuple(_adapt_callback(c) for c in callbacks)

    return callbacks


def _handle_nice_params(embedding: np.ndarray, optim_params: dict) -> None:
    """Convert the user-friendly params into something the optimizer can
    understand."""
    n_samples = embedding.shape[0]
    # Handle callbacks
    optim_params["callbacks"] = _check_callbacks(optim_params.get("callbacks"))
    optim_params["use_callbacks"] = optim_params["callbacks"] is not None

    # Handle negative gradient method
    negative_gradient_method = optim_params.pop("negative_gradient_method", "auto")
    if isinstance(negative_gradient_method, str) and negative_gradient_method == "auto":
        if n_samples < 10_000:
            negative_gradient_method = "bh"
        else:
            negative_gradient_method = "fft"
        log.info(
            f"Automatically determined negative gradient method `{negative_gradient_method}`"
        )

    if callable(negative_gradient_method):
        negative_gradient_method = negative_gradient_method
    elif negative_gradient_method in {"bh", "BH", "barnes-hut"}:
        negative_gradient_method = kl_divergence_bh
    elif negative_gradient_method in {"fft", "FFT", "interpolation"}:
        negative_gradient_method = kl_divergence_fft
    else:
        raise ValueError(
            "Unrecognized gradient method. Please choose one of "
            "the supported methods or provide a valid callback."
        )
    # `gradient_descent` uses the more informative name `objective_function`
    optim_params["objective_function"] = negative_gradient_method

    # Handle number of jobs
    n_jobs = optim_params.get("n_jobs", 1)
    if n_jobs < 0:
        n_cores = multiprocessing.cpu_count()
        # Add negative number of n_jobs to the number of cores, but increment by
        # one because -1 indicates using all cores, -2 all except one, and so on
        n_jobs = n_cores + n_jobs + 1

    # If the number of jobs, after this correction is still <= 0, then the user
    # probably thought they had more cores, so we'll default to 1
    if n_jobs <= 0:
        log.warning(
            "`n_jobs` receieved value %d but only %d cores are available. "
            "Defaulting to single job." % (optim_params["n_jobs"], n_cores)
        )
        n_jobs = 1

    optim_params["n_jobs"] = n_jobs

    # Determine learning rate if requested
    learning_rate = optim_params.get("learning_rate", "auto")
    if learning_rate == "auto":
        exaggeration = optim_params.get("exaggeration", None)
        if exaggeration is None:
            exaggeration = 1
        learning_rate = n_samples / exaggeration
    optim_params["learning_rate"] = learning_rate


def __check_init_num_samples(num_samples, required_num_samples):
    if num_samples != required_num_samples:
        raise ValueError(
            "The provided initialization contains a different number "
            "of points (%d) than the data provided (%d)."
            % (num_samples, required_num_samples)
        )


def __check_init_num_dimensions(num_dimensions, required_num_dimensions):
    if num_dimensions != required_num_dimensions:
        raise ValueError(
            "The provided initialization contains a different number "
            "of components (%d) than the embedding (%d)."
            % (num_dimensions, required_num_dimensions)
        )


init_checks = SimpleNamespace(
    num_samples=__check_init_num_samples,
    num_dimensions=__check_init_num_dimensions,
)


class OptimizationInterrupt(InterruptedError):
    """Optimization was interrupted by a callback.

    Parameters
    ----------
    error: float
        The KL divergence of the embedding.

    final_embedding: Union[TSNEEmbedding, PartialTSNEEmbedding]
        Is either a partial or full embedding, depending on where the error was
        raised.

    """

    def __init__(self, error, final_embedding):
        super().__init__()
        self.error = error
        self.final_embedding = final_embedding


class PartialTSNEEmbedding(np.ndarray):
    """A partial t-SNE embedding.

    A partial embedding is created when we take an existing
    :class:`TSNEEmbedding` and embed new samples into the embedding space. It
    differs from the typical embedding in that it is not possible to add new
    samples to a partial embedding and would generally be a bad idea.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    embedding: np.ndarray
        Initial positions for each data point.

    reference_embedding: TSNEEmbedding
        The embedding into which the new samples are to be added.

    P : array_like
        An :math:`N \\times M` affinity matrix containing the affinities from
        each new data point :math:`n` to each data point in the existing
        embedding :math:`m`.

    learning_rate: Union[str, float]
        The learning rate for t-SNE optimization. When ``learning_rate="auto"``
        the appropriate learning rate is selected according to N / exaggeration
        as determined in Belkina et al. (2019), Nature Communications. Note that
        this will result in a different learning rate during the early
        exaggeration phase and afterwards. This should *not* be used when
        adding samples into existing embeddings, where the learning rate often
        needs to be much lower to obtain convergence.

    exaggeration: float
        The exaggeration factor is used to increase the attractive forces of
        nearby points, producing more compact clusters.

    momentum: float
        Momentum accounts for gradient directions from previous iterations,
        resulting in faster convergence.

    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate and can be set
        using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
        For larger data sets, the FFT accelerated interpolation method is more
        appropriate and can be set using one of the following aliases: ``fft``,
        ``FFT`` or ``ìnterpolation``. Alternatively, you can use ``auto`` to
        approximately select the faster method.

    theta: float
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime.

    n_interpolation_points: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3.

    min_num_intervals: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The minimum number of grid cells to use, regardless of the
        ``ints_in_interval`` parameter. Higher values provide more accurate
        gradient estimations.

    random_state: Union[int, RandomState]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    callbacks: Callable[[IterationState], bool]
        Callbacks, which will be run every ``callbacks_every_iters`` iterations.
        Each callback receives an :class:`IterationState` snapshot containing
        the current iteration, embedding, gradient, error, exaggeration, and
        degrees-of-freedom values. Returning ``True`` from any callback will
        interrupt the optimization. The legacy three-argument signature
        ``callback(iteration, error, embedding)`` is still accepted but emits a
        ``FutureWarning`` and will be removed in a future release.

    callbacks_every_iters: int
        How many iterations should pass between each time the callbacks are
        invoked.

    optimizer: gradient_descent
        Optionally, an existing optimizer can be used for optimization. This is
        useful for keeping momentum gains between different calls to
        :func:`optimize`.

    Attributes
    ----------
    kl_divergence: float
        The KL divergence or error of the embedding.

    """

    def __new__(
        cls,
        embedding,
        reference_embedding,
        P,
        optimizer=None,
        **gradient_descent_params,
    ):
        init_checks.num_samples(embedding.shape[0], P.shape[0])

        obj = np.array(embedding, dtype=np.float64, order="C").view(
            PartialTSNEEmbedding
        )

        obj.reference_embedding = reference_embedding
        obj.P = P
        obj.gradient_descent_params = gradient_descent_params

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, gradient_descent):
            raise TypeError(
                "`optimizer` must be an instance of `%s`, but got `%s`."
                % (gradient_descent.__class__.__name__, type(optimizer))
            )
        obj.optimizer = optimizer

        obj.kl_divergence = None

        return obj

    def optimize(
        self,
        n_iter,
        inplace=False,
        propagate_exception=False,
        **gradient_descent_params,
    ):
        """Run optmization on the embedding for a given number of steps.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to N / exaggeration as determined in Belkina et al.
            (2019), Nature Communications. Note that this will result in a
            different learning rate during the early exaggeration phase and
            afterwards. This should *not* be used when adding samples into
            existing embeddings, where the learning rate often needs to be much
            lower to obtain convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        negative_gradient_method: str
            Specifies the negative gradient approximation method to use. For
            smaller data sets, the Barnes-Hut approximation is appropriate and
            can be set using one of the following aliases: ``bh``, ``BH`` or
            ``barnes-hut``. For larger data sets, the FFT accelerated
            interpolation method is more appropriate and can be set using one of
            the following aliases: ``fft``, ``FFT`` or ``ìnterpolation``.
            Alternatively, you can use ``auto`` to approximately select the
            faster method.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.

        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        callbacks: Callable[[IterationState], bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations. Each callback receives an :class:`IterationState`
            snapshot containing the current iteration, embedding, gradient,
            error, exaggeration, and degrees-of-freedom values. Returning
            ``True`` from any callback will interrupt the optimization.
            The legacy three-argument signature
            ``callback(iteration, error, embedding)`` is still accepted but
            emits a ``FutureWarning`` and will be removed in a future release.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        PartialTSNEEmbedding
            An optimized partial t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = PartialTSNEEmbedding(
                np.copy(self),
                self.reference_embedding,
                self.P,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        optim_params["n_iter"] = n_iter
        _handle_nice_params(embedding, optim_params)

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding,
                reference_embedding=self.reference_embedding,
                P=self.P,
                **optim_params,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding


class TSNEEmbedding(np.ndarray):
    """A t-SNE embedding.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    embedding: np.ndarray
        Initial positions for each data point.

    affinities: Affinities
        An affinity index which can be used to compute the affinities of new
        points to the points in the existing embedding. The affinity index also
        contains the affinity matrix :math:`P` used during optimization.

    learning_rate: Union[str, float]
        The learning rate for t-SNE optimization. When ``learning_rate="auto"``
        the appropriate learning rate is selected according to N / exaggeration
        as determined in Belkina et al. (2019), Nature Communications. Note that
        this will result in a different learning rate during the early
        exaggeration phase and afterwards. This should *not* be used when
        adding samples into existing embeddings, where the learning rate often
        needs to be much lower to obtain convergence.

    exaggeration: float
        The exaggeration factor is used to increase the attractive forces of
        nearby points, producing more compact clusters.

    dof: float
        Degrees of freedom as described in Kobak et al. "Heavy-tailed kernels
        reveal a finer cluster structure in t-SNE visualisations", 2019.

    momentum: float
        Momentum accounts for gradient directions from previous iterations,
        resulting in faster convergence.

    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate and can be set
        using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
        For larger data sets, the FFT accelerated interpolation method is more
        appropriate and can be set using one of the following aliases: ``fft``,
        ``FFT`` or ``ìnterpolation``.A lternatively, you can use ``auto`` to
        approximately select the faster method.

    theta: float
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime.

    n_interpolation_points: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3.

    min_num_intervals: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The minimum number of grid cells to use, regardless of the
        ``ints_in_interval`` parameter. Higher values provide more accurate
        gradient estimations.

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    random_state: Union[int, RandomState]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    callbacks: Callable[[IterationState], bool]
        Callbacks, which will be run every ``callbacks_every_iters`` iterations.
        Each callback receives an :class:`IterationState` snapshot containing
        the current iteration, embedding, gradient, error, exaggeration, and
        degrees-of-freedom values. Returning ``True`` from any callback will
        interrupt the optimization. The legacy three-argument signature
        ``callback(iteration, error, embedding)`` is still accepted but emits a
        ``FutureWarning`` and will be removed in a future release.

    callbacks_every_iters: int
        How many iterations should pass between each time the callbacks are
        invoked.

    optimizer: gradient_descent
        Optionally, an existing optimizer can be used for optimization. This is
        useful for keeping momentum gains between different calls to
        :func:`optimize`.

    Attributes
    ----------
    kl_divergence: float
        The KL divergence or error of the embedding.

    """

    def __new__(
        cls,
        embedding,
        affinities,
        dof=1,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        negative_gradient_method="auto",
        random_state=None,
        optimizer=None,
        **gradient_descent_params,
    ):
        init_checks.num_samples(embedding.shape[0], affinities.P.shape[0])

        obj = np.array(embedding, dtype=np.float64, order="C").view(TSNEEmbedding)

        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.gradient_descent_params.update(
            {
                "negative_gradient_method": negative_gradient_method,
                "n_interpolation_points": n_interpolation_points,
                "min_num_intervals": min_num_intervals,
                "ints_in_interval": ints_in_interval,
                "dof": dof,
            }
        )
        obj.random_state = random_state

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, gradient_descent):
            raise TypeError(
                "`optimizer` must be an instance of `%s`, but got `%s`."
                % (gradient_descent.__class__.__name__, type(optimizer))
            )
        obj.optimizer = optimizer

        obj.kl_divergence = None

        # Interpolation grid variables
        obj.interp_coeffs = None
        obj.box_x_lower_bounds = None
        obj.box_y_lower_bounds = None
        return obj

    def optimize(
        self,
        n_iter,
        inplace=False,
        propagate_exception=False,
        **gradient_descent_params,
    ):
        """Run optmization on the embedding for a given number of steps.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to N / exaggeration as determined in Belkina et al.
            (2019), Nature Communications. Note that this will result in a
            different learning rate during the early exaggeration phase and
            afterwards. This should *not* be used when adding samples into
            existing embeddings, where the learning rate often needs to be much
            lower to obtain convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        dof: float
            Degrees of freedom as described in Kobak et al. "Heavy-tailed kernels
            reveal a finer cluster structure in t-SNE visualisations", 2019.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        negative_gradient_method: str
            Specifies the negative gradient approximation method to use. For
            smaller data sets, the Barnes-Hut approximation is appropriate and
            can be set using one of the following aliases: ``bh``, ``BH`` or
            ``barnes-hut``. For larger data sets, the FFT accelerated
            interpolation method is more appropriate and can be set using one of
            the following aliases: ``fft``, ``FFT`` or ``ìnterpolation``.
            Alternatively, you can use ``auto`` to approximately select the
            faster method.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.

        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.

        max_grad_norm: float
            Maximum gradient norm. If the norm exceeds this value, it will be
            clipped. This is most beneficial when adding points into an existing
            embedding and the new points overlap with the reference points,
            leading to large gradients. This can make points "shoot off" from
            the embedding, causing the interpolation method to compute a very
            large grid, and leads to worse results.

        max_step_norm: float
            Maximum update norm. If the norm exceeds this value, it will be
            clipped. This prevents points from "shooting off" from
            the embedding.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        callbacks: Callable[[IterationState], bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations. Each callback receives an :class:`IterationState`
            snapshot containing the current iteration, embedding, gradient,
            error, exaggeration, and degrees-of-freedom values. Returning
            ``True`` from any callback will interrupt the optimization.
            The legacy three-argument signature
            ``callback(iteration, error, embedding)`` is still accepted but
            emits a ``FutureWarning`` and will be removed in a future release.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        TSNEEmbedding
            An optimized t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = TSNEEmbedding(
                np.copy(self),
                self.affinities,
                random_state=self.random_state,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        optim_params["n_iter"] = n_iter
        _handle_nice_params(embedding, optim_params)

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding, P=self.affinities.P, **optim_params
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding

    def transform(
        self,
        X,
        perplexity=5,
        initialization="median",
        k=25,
        learning_rate=0.1,
        early_exaggeration=4,
        early_exaggeration_iter=0,
        exaggeration=1.5,
        n_iter=250,
        initial_momentum=0.8,
        final_momentum=0.8,
        max_grad_norm=0.25,
        max_step_norm=None,
    ):
        """Embed new points into the existing embedding.

        This procedure optimizes each point only with respect to the existing
        embedding i.e. it ignores any interactions between the points in ``X``
        among themselves.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be added to the existing embedding.

        perplexity: float
            Perplexity can be thought of as the continuous :math:`k` number of
            nearest neighbors, for which t-SNE will attempt to preserve
            distances. However, when transforming, we only consider neighbors in
            the existing embedding i.e. each data point is placed into the
            embedding, independently of other new data points.

        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``median``, ``weighted`` or
            ``random``. In all cases, ``median`` of ``weighted`` should be
            preferred.

        k: int
            The number of nearest neighbors to consider when initially placing
            the point onto the embedding. This is different from ``perpelxity``
            because perplexity affects optimization while this only affects the
            initial point positions.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to N / exaggeration as determined in Belkina et al.
            (2019), Nature Communications. Note that this will result in a
            different learning rate during the early exaggeration phase and
            afterwards. This should *not* be used when adding samples into
            existing embeddings, where the learning rate often needs to be much
            lower to obtain convergence.

        early_exaggeration_iter: int
            The number of iterations to run in the *early exaggeration* phase.

        early_exaggeration: float
            The exaggeration factor to use during the *early exaggeration* phase.
            Typical values range from 4 to 32.

        n_iter: int
            The number of iterations to run in the normal optimization regime.

        exaggeration: float
            The exaggeration factor to use during the normal optimization phase.
            This can be used to form more densely packed clusters and is useful
            for large data sets.

        initial_momentum: float
            The momentum to use during the *early exaggeration* phase.

        final_momentum: float
            The momentum to use during the normal optimization phase.

        max_grad_norm: float
            Maximum gradient norm. If the norm exceeds this value, it will be
            clipped. This is most beneficial when adding points into an existing
            embedding and the new points overlap with the reference points,
            leading to large gradients. This can make points "shoot off" from
            the embedding, causing the interpolation method to compute a very
            large grid, and leads to worse results.

        max_step_norm: float
            Maximum update norm. If the norm exceeds this value, it will be
            clipped. This prevents points from "shooting off" from
            the embedding.

        Returns
        -------
        PartialTSNEEmbedding
            The positions of the new points in the embedding space.

        """

        # Since the standard usage of t-SNE uses perplexities, check if the
        # currently used affinity class supports either the `perplexity` or
        # `perplexities` parameters.
        affinity_signature = inspect.signature(self.affinities.to_new)
        if "perplexity" in affinity_signature.parameters:
            affinity_params = {"perplexity": perplexity}
        elif "perplexities" in affinity_signature.parameters:
            affinity_params = {"perplexities": perplexity}
        else:
            raise TypeError(
                f"`transform` currently does not support "
                f"{self.affinities.__class__.__name__} affinities. Please use "
                f"`prepare_partial` and `optimize` to add points to the embedding."
            )

        # Center the current embedding
        self -= (np.max(self, axis=0) + np.min(self, axis=0)) / 2

        embedding = self.prepare_partial(
            X, initialization=initialization, k=k, **affinity_params
        )

        try:
            embedding.optimize(
                n_iter=early_exaggeration_iter,
                learning_rate=learning_rate,
                exaggeration=early_exaggeration,
                momentum=initial_momentum,
                inplace=True,
                propagate_exception=True,
                max_grad_norm=max_grad_norm,
                max_step_norm=max_step_norm,
            )
            embedding.optimize(
                n_iter=n_iter,
                learning_rate=learning_rate,
                exaggeration=exaggeration,
                momentum=final_momentum,
                inplace=True,
                propagate_exception=True,
                max_grad_norm=max_grad_norm,
                max_step_norm=max_step_norm,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            embedding = ex.final_embedding

        return embedding

    def prepare_partial(self, X, initialization="median", k=25, **affinity_params):
        """Prepare a partial embedding which can be optimized.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be added to the existing embedding.

        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``median``, ``weighted`` or
            ``random``. In all cases, ``median`` of ``weighted`` should be
            preferred.

        k: int
            The number of nearest neighbors to consider when initially placing
            the point onto the embedding. This is different from ``perpelxity``
            because perplexity affects optimization while this only affects the
            initial point positions.

        **affinity_params: dict
            Additional params to be passed to the ``Affinities.to_new`` method.
            Please see individual :class:`~openTSNE.affinity.Affinities`
            implementations as the parameters differ between implementations.

        Returns
        -------
        PartialTSNEEmbedding
            An unoptimized :class:`PartialTSNEEmbedding` object, prepared for
            optimization.

        """
        # To maintain perfect backwards compatibility and to handle the very
        # specific case when the user wants to pass in `perplexity` to the
        # multiscale affinity object (multiscale accepts `perplexities`), rename
        # this parameter so everything works
        affinity_signature = inspect.signature(self.affinities.to_new)
        if (
            "perplexities" in affinity_signature.parameters
            and "perplexities" in affinity_signature.parameters
            and "perplexity" in affinity_params
            and "perplexities" not in affinity_params
        ):
            affinity_params["perplexities"] = affinity_params.pop("perplexity")

        P, neighbors, distances = self.affinities.to_new(
            X, return_distances=True, **affinity_params
        )

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            init_checks.num_samples(initialization.shape[0], X.shape[0])
            init_checks.num_dimensions(initialization.shape[1], self.shape[1])

            embedding = np.array(initialization)
        # Random initialization with isotropic normal distribution
        elif initialization == "random":
            embedding = initialization_scheme.random(
                X, self.shape[1], self.random_state
            )
        elif initialization == "weighted":
            embedding = initialization_scheme.weighted_mean(
                X, self, neighbors[:, :k], distances[:, :k]
            )
        elif initialization == "median":
            embedding = initialization_scheme.median(self, neighbors[:, :k])
        else:
            raise ValueError(f"Unrecognized initialization scheme `{initialization}`.")

        return PartialTSNEEmbedding(
            embedding,
            self,
            P=P,
            **self.gradient_descent_params,
        )

    def prepare_interpolation_grid(self, padding=0.25):
        """Evaluate and save the interpolation grid coefficients.

        Parameters
        ----------
        padding: float
            During standard optimization, the grid hugs the embedding points as
            closely as possible, but this is not what we want when performing
            transform. This paraemter specifies how much empty space should be
            appended in each dimension. The values are given in percentages.

        """
        # Center embedding into our grid
        self -= (np.max(self, axis=0) + np.min(self, axis=0)) / 2

        if self.shape[1] == 1:
            f = _tsne.prepare_negative_gradient_fft_interpolation_grid_1d
        elif self.shape[1] == 2:
            f = _tsne.prepare_negative_gradient_fft_interpolation_grid_2d
        else:
            raise RuntimeError("Cannot prepare interpolation grid for >2d embeddings")

        result = f(
            self.ravel() if self.shape[1] == 1 else self,
            self.gradient_descent_params["n_interpolation_points"],
            self.gradient_descent_params["min_num_intervals"],
            self.gradient_descent_params["ints_in_interval"],
            self.gradient_descent_params["dof"],
            padding=padding,
        )

        if len(result) == 2:  # 1d case
            self.interp_coeffs, self.box_x_lower_bounds = result
        elif len(result) == 3:  # 2d case
            self.interp_coeffs, self.box_x_lower_bounds, self.box_y_lower_bounds = (
                result
            )
        else:
            raise RuntimeError(
                "Prepare interpolation grid function returned >3 values!"
            )

    def __reduce__(self):
        state = super().__reduce__()
        new_state = state[2] + (
            self.optimizer,
            self.affinities,
            self.gradient_descent_params,
            self.random_state,
            self.kl_divergence,
            self.interp_coeffs,
            self.box_x_lower_bounds,
            self.box_y_lower_bounds,
        )
        return state[0], state[1], new_state

    def __setstate__(self, state):
        self.box_y_lower_bounds = state[-1]
        self.box_x_lower_bounds = state[-2]
        self.interp_coeffs = state[-3]
        self.kl_divergence = state[-4]
        self.random_state = state[-5]
        self.gradient_descent_params = state[-6]
        self.affinities = state[-7]

        if len(state) == 12:  # backwards compat (when I forgot optimizer)
            self.optimizer = gradient_descent()
            super().__setstate__(state[:-7])
        else:
            self.optimizer = state[-8]
            super().__setstate__(state[:-8])


class TSNE(BaseEstimator):
    """t-Distributed Stochastic Neighbor Embedding.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    n_components: int
        The dimension of the embedding space. This deafults to 2 for easy
        visualization, but sometimes 1 is used for t-SNE heatmaps. t-SNE is
        not designed to embed into higher dimension and please note that
        acceleration schemes break down and are not fully implemented.

    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        nearest neighbors, for which t-SNE will attempt to preserve distances.

    learning_rate: Union[str, float]
        The learning rate for t-SNE optimization. When ``learning_rate="auto"``
        the appropriate learning rate is selected according to N / exaggeration
        as determined in Belkina et al. (2019), Nature Communications. Note that
        this will result in a different learning rate during the early
        exaggeration phase and afterwards. This should *not* be used when
        adding samples into existing embeddings, where the learning rate often
        needs to be much lower to obtain convergence.

    early_exaggeration_iter: int
        The number of iterations to run in the *early exaggeration* phase.

    early_exaggeration: Union[str, float]
        The exaggeration factor to use during the *early exaggeration* phase.
        Typical values range from 4 to 32. When ``early_exaggeration="auto"``
        early exaggeration factor defaults to 12, unless desired subsequent
        exaggeration is higher, i.e.: ``early_exaggeration = max(12,
        exaggeration)``.

    n_iter: int
        The number of iterations to run in the normal optimization regime.

    exaggeration: float
        The exaggeration factor to use during the normal optimization phase.
        This can be used to form more densely packed clusters and is useful
        for large data sets.

    dof: float
        Degrees of freedom as described in Kobak et al. "Heavy-tailed kernels
        reveal a finer cluster structure in t-SNE visualisations", 2019.

    theta: float
        Only used when ``negative_gradient_method="bh"`` or its other aliases.
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime. Default is 0.5.

    n_interpolation_points: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3.

    min_num_intervals: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The minimum number of grid cells to use, regardless of the
        ``ints_in_interval`` parameter. Higher values provide more accurate
        gradient estimations.

    ints_in_interval: float
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        Indicates how large a grid cell should be e.g. a value of 3 indicates a
        grid side length of 3. Lower values provide more accurate gradient
        estimations.

    initialization: Union[np.ndarray, str]
        The initial point positions to be used in the embedding space. Can be a
        precomputed numpy array, ``pca``, ``spectral`` or ``random``. Please
        note that when passing in a precomputed positions, it is highly
        recommended that the point positions have small variance
        (std(Y) < 0.0001), otherwise you may get poor embeddings.

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    initial_momentum: float
        The momentum to use during the *early exaggeration* phase.

    final_momentum: float
        The momentum to use during the normal optimization phase.

    max_grad_norm: float
        Maximum gradient norm. If the norm exceeds this value, it will be
        clipped. This is most beneficial when adding points into an existing
        embedding and the new points overlap with the reference points,
        leading to large gradients. This can make points "shoot off" from
        the embedding, causing the interpolation method to compute a very
        large grid, and leads to worse results.

    max_step_norm: float
        Maximum update norm. If the norm exceeds this value, it will be
        clipped. This prevents points from "shooting off" from
        the embedding.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    neighbors: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
        if the input data matrix is not a sparse object and if Annoy supports
        the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
        nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.
        ``hnsw`` uses hnswlib, which must be installed separately
        (``pip install hnswlib``) and supports the ``euclidean``, ``l2``,
        ``cosine``, ``dot``, and ``ip`` metrics.

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate and can be set
        using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
        For larger data sets, the FFT accelerated interpolation method is more
        appropriate and can be set using one of the following aliases: ``fft``,
        ``FFT`` or ``ìnterpolation``. Alternatively, you can use ``auto`` to
        approximately select the faster method.

    callbacks: Union[Callable, List[Callable]]
        Callbacks, which will be run every ``callbacks_every_iters`` iterations.

    callbacks_every_iters: int
        How many iterations should pass between each time the callbacks are
        invoked.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    """

    def __init__(
        self,
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        early_exaggeration_iter=250,
        early_exaggeration="auto",
        n_iter=500,
        exaggeration=None,
        dof=1,
        initial_dof=None,
        theta=0.5,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        initialization="pca",
        metric="euclidean",
        metric_params=None,
        initial_momentum=0.8,
        final_momentum=0.8,
        max_grad_norm=None,
        max_step_norm=5,
        n_jobs=1,
        neighbors="auto",
        knn_kwargs=None,
        negative_gradient_method="auto",
        callbacks=None,
        callbacks_every_iters=50,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        if early_exaggeration == "auto":
            if exaggeration is None:
                self.early_exaggeration = 12
            else:
                self.early_exaggeration = max(12, exaggeration)
        else:
            self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.n_iter = n_iter
        self.exaggeration = exaggeration
        self.dof = dof
        self.initial_dof = initial_dof
        self.theta = theta
        self.n_interpolation_points = n_interpolation_points
        self.min_num_intervals = min_num_intervals
        self.ints_in_interval = ints_in_interval

        # Check if the number of components match the initialization dimension
        if isinstance(initialization, np.ndarray):
            init_checks.num_dimensions(initialization.shape[1], n_components)
        self.initialization = initialization

        self.metric = metric
        self.metric_params = metric_params
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.max_grad_norm = max_grad_norm
        self.max_step_norm = max_step_norm
        self.n_jobs = n_jobs

        self.neighbors = neighbors
        self.knn_kwargs = knn_kwargs
        self.negative_gradient_method = negative_gradient_method

        self.callbacks = callbacks
        self.callbacks_every_iters = callbacks_every_iters

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X=None, affinities=None, initialization=None):
        """Fit a t-SNE embedding for a given data set.

        Runs the standard t-SNE optimization, consisting of the early
        exaggeration phase and a normal optimization phase.

        This function call be called in two ways.

        1.  We can call it in the standard way using a ``np.array``. This will
            compute the affinity matrix and initialization, and run the optimization
            as usual.
        2.  We can also pass in a precomputed ``affinity`` object, which will
            override the affinity-related paramters specified in the constructor.
            This is useful when you wish to use custom affinity objects.

        Please note that some initialization schemes require ``X`` to be specified,
        e.g. PCA. If the initilization is not able to be computed, we default to
        using spectral initilization calculated from the affinity matrix.

        Parameters
        ----------
        X: Optional[np.ndarray]
            The data matrix to be embedded.

        affinities: Optional[openTSNE.affinity.Affinities]
            A precomputed affinity object. If specified, other affinity-related
            parameters are ignored e.g. `perplexity` and anything nearest-neighbor
            search related.

        initialization: Optional[np.ndarray]
            The initial point positions to be used in the embedding space. Can be
            a precomputed numpy array, ``pca``, ``spectral`` or ``random``. Please
            note that when passing in a precomputed positions, it is highly
            recommended that the point positions have small variance
            (std(Y) < 0.0001), otherwise you may get poor embeddings.

        Returns
        -------
        TSNEEmbedding
            A fully optimized t-SNE embedding.

        """
        if self.verbose:
            print("-" * 80, repr(self), "-" * 80, sep="\n")

        embedding = self.prepare_initial(X, affinities, initialization)

        try:
            # Early exaggeration with lower momentum to allow points to find more
            # easily move around and find their neighbors
            embedding.optimize(
                n_iter=self.early_exaggeration_iter,
                exaggeration=self.early_exaggeration,
                momentum=self.initial_momentum,
                inplace=True,
                propagate_exception=True,
            )

            # Restore actual affinity probabilities and increase momentum to get
            # final, optimized embedding
            embedding.optimize(
                n_iter=self.n_iter,
                exaggeration=self.exaggeration,
                momentum=self.final_momentum,
                inplace=True,
                propagate_exception=True,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            embedding = ex.final_embedding

        return embedding

    def prepare_initial(self, X=None, affinities=None, initialization=None):
        """Prepare the initial embedding which can be optimized as needed.

        This function call be called in two ways.

        1.  We can call it in the standard way using a ``np.array``. This will
            compute the affinity matrix and initialization as usual.
        2.  We can also pass in a precomputed ``affinity`` object, which will
            override the affinity-related paramters specified in the constructor.
            This is useful when you wish to use custom affinity objects.

        Please note that some initialization schemes require ``X`` to be specified,
        e.g. PCA. If the initilization is not able to be computed, we default to
        using spectral initilization calculated from the affinity matrix.

        Parameters
        ----------
        X: Optional[np.ndarray]
            The data matrix to be embedded.

        affinities: Optional[openTSNE.affinity.Affinities]
            A precomputed affinity object. If specified, other affinity-related
            parameters are ignored e.g. `perplexity` and anything nearest-neighbor
            search related.

        initialization: Optional[np.ndarray]
            The initial point positions to be used in the embedding space. Can be
            a precomputed numpy array, ``pca``, ``spectral`` or ``random``. Please
            note that when passing in a precomputed positions, it is highly
            recommended that the point positions have small variance
            (std(Y) < 0.0001), otherwise you may get poor embeddings.

        Returns
        -------
        TSNEEmbedding
            An unoptimized :class:`TSNEEmbedding` object, prepared for
            optimization.

        """
        # Either `X` or `affinities` must be specified
        if X is None and affinities is None and initialization is None:
            raise ValueError(
                "At least one of the parameters `X` or `affinities` must be specified!"
            )

        # If precomputed affinites are given, use those, otherwise proceed with
        # standard perpelxity-based affinites
        if affinities is None:
            affinities = MultiscaleMixture(
                X,
                self.perplexity,
                method=self.neighbors,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                knn_kwargs=self.knn_kwargs,
            )
        else:
            if not isinstance(affinities, Affinities):
                raise ValueError(
                    "`affinities` must be an instance of `openTSNE.affinity.Affinities`"
                )
            log.info(
                "Precomputed affinities provided. Ignoring perplexity-related parameters."
            )

        # If a precomputed initialization was specified, use that, otherwise
        # use the parameters specified in the constructor
        if initialization is None:
            initialization = self.initialization
            log.info(
                "Precomputed initialization provided. Ignoring initalization-related "
                "parameters."
            )

        # If only the affinites have been specified, and the initialization depends
        # on `X`, switch to spectral initalization
        if X is None and isinstance(initialization, str) and initialization == "pca":
            log.warning(
                "Attempting to use `pca` initalization, but no `X` matrix specified! "
                "Using `spectral` initilization instead, which doesn't need access "
                "to the data matrix"
            )
            initialization = "spectral"

        # Same spiel for precomputed distance matrices
        if (
            self.metric == "precomputed"
            and isinstance(initialization, str)
            and initialization == "pca"
        ):
            log.warning(
                "Attempting to use `pca` initalization, but using precomputed "
                "distance matrix! Using `spectral` initilization instead, which "
                "doesn't need access to the data matrix."
            )
            initialization = "spectral"

        # Determine the number of samples in the input data set
        if X is not None:
            n_samples = X.shape[0]
        else:
            n_samples = affinities.P.shape[0]

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            init_checks.num_samples(initialization.shape[0], n_samples)
            init_checks.num_dimensions(initialization.shape[1], self.n_components)

            embedding = np.array(initialization)

            stddev = np.std(embedding, axis=0)
            if any(stddev > 1e-2):
                log.warning(
                    "Standard deviation of embedding is greater than 0.0001. Initial "
                    "embeddings with high variance may have display poor convergence."
                )

        elif initialization == "pca":
            embedding = initialization_scheme.pca(
                X,
                self.n_components,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        elif initialization == "random":
            embedding = initialization_scheme.random(
                n_samples,
                self.n_components,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        elif initialization == "spectral":
            embedding = initialization_scheme.spectral(
                affinities.P,
                self.n_components,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unrecognized initialization scheme `{initialization}`.")

        gradient_descent_params = {
            "dof": self.dof,
            "negative_gradient_method": self.negative_gradient_method,
            "learning_rate": self.learning_rate,
            # By default, use the momentum used in unexaggerated phase
            "momentum": self.final_momentum,
            # Barnes-Hut params
            "theta": self.theta,
            # Interpolation params
            "n_interpolation_points": self.n_interpolation_points,
            "min_num_intervals": self.min_num_intervals,
            "ints_in_interval": self.ints_in_interval,
            "max_grad_norm": self.max_grad_norm,
            "max_step_norm": self.max_step_norm,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            # Callback params
            "callbacks": self.callbacks,
            "callbacks_every_iters": self.callbacks_every_iters,
            "initial_dof": self.initial_dof,
        }

        return TSNEEmbedding(
            embedding,
            affinities=affinities,
            random_state=self.random_state,
            **gradient_descent_params,
        )


def kl_divergence_bh(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    should_eval_error=False,
    n_jobs=1,
    compute_dof_grad=False,
    **_,
):
    if embedding.ndim != 1 and embedding.shape[1] > 3:
        warnings.warn(
            "BH t-SNE for >3 dimensions can lead to segfaults and is generally "
            "a bad idea. In the future, calling BH with >3 dimensions will "
            "raise a RuntimeError",
            category=FutureWarning,
        )
        # raise RuntimeError(
        #     "BH t-SNE for >3 dimensions is currently unsupported (and "
        #     "generally a bad idea)"
        # )

    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself. We've also got to make sure that the points'
    # interactions don't interfere with each other
    pairwise_normalization = reference_embedding is None
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q, alpha_grad_neg = _tsne.estimate_negative_gradient_bh(
        tree,
        embedding,
        gradient,
        **bh_params,
        dof=dof,
        num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
        compute_dof_grad=compute_dof_grad,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_, alpha_grad_pos = _tsne.estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof=dof,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
        compute_dof_grad=compute_dof_grad,
    )

    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    dof_grad = alpha_grad_pos - alpha_grad_neg if compute_dof_grad else 0.0
    return GradientResult(
        error=kl_divergence_,
        gradient=gradient,
        dof_grad=dof_grad,
    )


def kl_divergence_fft(
    embedding,
    P,
    dof,
    fft_params,
    reference_embedding=None,
    should_eval_error=False,
    n_jobs=1,
    **_,
):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # Compute negative gradient.
    if embedding.ndim == 1 or embedding.shape[1] == 1:
        if reference_embedding is not None:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d_with_grid(
                embedding.ravel(),
                gradient.ravel(),
                reference_embedding.interp_coeffs,
                reference_embedding.box_x_lower_bounds,
                fft_params["n_interpolation_points"],
                dof=dof,
            )
        else:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d(
                embedding.ravel(), gradient.ravel(), **fft_params, dof=dof,
                num_threads=n_jobs,
            )
    elif embedding.shape[1] == 2:
        if reference_embedding is not None:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d_with_grid(
                embedding,
                gradient,
                reference_embedding.interp_coeffs,
                reference_embedding.box_x_lower_bounds,
                reference_embedding.box_y_lower_bounds,
                fft_params["n_interpolation_points"],
                dof=dof,
            )
        else:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d(
                embedding, gradient, **fft_params, dof=dof, num_threads=n_jobs,
            )
    else:
        raise RuntimeError(
            "Interpolation based t-SNE for >2 dimensions is currently "
            "unsupported (and generally a bad idea)"
        )

    # The positive gradient function needs a reference embedding always
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute positive gradient
    sum_P, kl_divergence_, _alpha_grad_pos = _tsne.estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
    )

    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    # The FFT negative-gradient kernels do not compute the dof-gradient term,
    # so we cannot return a meaningful value here. Learning dof requires the
    # Barnes-Hut objective; see the guard in `gradient_descent.__call__`.
    return GradientResult(error=kl_divergence_, gradient=gradient, dof_grad=0.0)


class gradient_descent:
    def __init__(self):
        self.gains = None
        self.update = None
        self.dof_update = 0.0
        self.dof_gain = 1.0

    def copy(self):
        optimizer = self.__class__()
        if self.gains is not None:
            optimizer.gains = np.copy(self.gains)
        if self.update is not None:
            optimizer.update = np.copy(self.update)
        return optimizer

    def __call__(
        self,
        embedding,
        P,
        n_iter,
        objective_function,
        learning_rate="auto",
        momentum=0.8,
        exaggeration=None,
        dof=1,
        initial_dof=None,
        min_gain=0.01,
        max_grad_norm=None,
        max_step_norm=5,
        theta=0.5,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        reference_embedding=None,
        n_jobs=1,
        use_callbacks=False,
        callbacks=None,
        callbacks_every_iters=50,
        eval_error_every_iter=50,
        verbose=False,
    ):
        """Perform batch gradient descent with momentum and gains.

        Parameters
        ----------
        embedding: np.ndarray
            The embedding :math:`Y`.

        P: array_like
            Joint probability matrix :math:`P`.

        n_iter: int
            The number of iterations to run for.

        objective_function: Callable[..., Tuple[float, np.ndarray]]
            A callable that evaluates the error and gradient for the current
            embedding.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to N / exaggeration as determined in Belkina et al.
            (2019), Nature Communications. Note that this will result in a
            different learning rate during the early exaggeration phase and
            afterwards. This should *not* be used when adding samples into
            existing embeddings, where the learning rate often needs to be much
            lower to obtain convergence.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        dof: float
            Degrees of freedom of the Student's t-distribution.

        optimize_for_alpha: bool
            If True, perform the optimization for the alpha via gradient descent.
            **Implemented only for the Barnes-Hut objective function.**

        min_gain: float
            Minimum individual gain for each parameter.

        max_grad_norm: float
            Maximum gradient norm. If the norm exceeds this value, it will be
            clipped. This is most beneficial when adding points into an existing
            embedding and the new points overlap with the reference points,
            leading to large gradients. This can make points "shoot off" from
            the embedding, causing the interpolation method to compute a very
            large grid, and leads to worse results.

        max_step_norm: float
            Maximum update norm. If the norm exceeds this value, it will be
            clipped. This prevents points from "shooting off" from
            the embedding.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        ints_in_interval: float
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. Indicates how large a grid cell should be e.g. a value of 3
            indicates a grid side length of 3. Lower values provide more
            accurate gradient estimations.

        reference_embedding: np.ndarray
            If we are adding points to an existing embedding, we have to compute
            the gradients and errors w.r.t. the existing embedding.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        use_callbacks: bool

        callbacks: Callable[[IterationState], bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations. Each callback receives an :class:`IterationState`
            snapshot containing the current iteration, embedding, gradient,
            error, exaggeration, and degrees-of-freedom values. Returning
            ``True`` from any callback will interrupt the optimization.
            The legacy three-argument signature
            ``callback(iteration, error, embedding)`` is still accepted but
            emits a ``FutureWarning`` and will be removed in a future release.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.
        eval_error_every_iter: int
            How often should the error be evaluated, by default 50

        Returns
        -------
        float
            The KL divergence of the optimized embedding.
        np.ndarray
            The optimized embedding Y.

        Raises
        ------
        OptimizationInterrupt
            If the provided callback interrupts the optimization, this is raised.

        """
        assert isinstance(embedding, np.ndarray), (
            "`embedding` must be an instance of `np.ndarray`. Got `%s` instead"
            % type(embedding)
        )

        if reference_embedding is not None:
            assert isinstance(reference_embedding, np.ndarray), (
                "`reference_embedding` must be an instance of `np.ndarray`. Got "
                "`%s` instead" % type(reference_embedding)
            )

        # If the interpolation grid has not yet been evaluated, do it now
        if (
            reference_embedding is not None
            and reference_embedding.interp_coeffs is None
            and objective_function is kl_divergence_fft
        ):
            reference_embedding.prepare_interpolation_grid()

        # If we're running transform and using the interpolation scheme, then we
        # should limit the range where new points can go to
        should_limit_range = False
        if reference_embedding is not None:
            if reference_embedding.box_x_lower_bounds is not None:
                should_limit_range = True
                lower_limit = reference_embedding.box_x_lower_bounds[0]
                upper_limit = reference_embedding.box_x_lower_bounds[-1]

        if self.update is None:
            self.update = np.zeros_like(embedding).view(np.ndarray)
        if self.gains is None:
            self.gains = np.ones_like(embedding).view(np.ndarray)

        bh_params = {"theta": theta}
        fft_params = {
            "n_interpolation_points": n_interpolation_points,
            "min_num_intervals": min_num_intervals,
            "ints_in_interval": ints_in_interval,
        }

        # Lie about the P values for bigger attraction forces
        if exaggeration is None:
            exaggeration = 1

        if exaggeration != 1:
            P *= exaggeration

        # Notify the callbacks that the optimization is about to start
        if isinstance(callbacks, Iterable):
            for callback in callbacks:
                # Only call function if present on object
                getattr(callback, "optimization_about_to_start", lambda: ...)()

        timer = utils.Timer(
            "Running optimization with exaggeration=%.2f, lr=%.2f for %d iterations..."
            % (exaggeration, learning_rate, n_iter),
            verbose=verbose,
        )
        timer.__enter__()

        if verbose:
            start_time = time()

        if dof == "auto":
            dof_ = 1.0 if initial_dof is None else initial_dof
            compute_dof_grad = True
        else:
            if initial_dof is not None:
                warnings.warn(
                    "`initial_dof=%s` is ignored because `dof=%s` is fixed; "
                    "`initial_dof` only applies when `dof='auto'`."
                    % (initial_dof, dof),
                    stacklevel=2,
                )
            dof_ = dof
            compute_dof_grad = False

        if dof == "auto" and objective_function is kl_divergence_fft:
            log.warning(
                "Learning the degrees of freedom (`dof='auto'`) is only "
                "implemented for the Barnes-Hut objective. The FFT objective "
                "does not compute the dof gradient, so dof will remain fixed "
                "at `initial_dof=%s`. Set `negative_gradient_method='bh'` to "
                "actually learn dof.",
                dof_,
            )

        for iteration in range(n_iter):
            should_call_callback = (
                use_callbacks and (iteration + 1) % callbacks_every_iters == 0
            )

            should_eval_error = should_call_callback or (
                verbose and (iteration + 1) % eval_error_every_iter == 0
            )

            result = objective_function(
                embedding,
                P,
                dof=dof_,
                bh_params=bh_params,
                fft_params=fft_params,
                reference_embedding=reference_embedding,
                n_jobs=n_jobs,
                should_eval_error=should_eval_error,
                compute_dof_grad=compute_dof_grad,
            )
            error = result.error
            gradient = result.gradient
            dof_grad = result.dof_grad

            # Snapshot the raw gradient before any in-place modification so the
            # callback sees what the objective actually returned.
            gradient_snapshot = gradient.copy() if should_call_callback else None

            # Clip gradients to avoid points shooting off. This can be an issue
            # when applying transform and points are initialized so that the new
            # points overlap with the reference points, leading to large
            # gradients
            if max_grad_norm is not None:
                norm = np.linalg.norm(gradient, axis=1)
                coeff = max_grad_norm / (norm + 1e-6)
                mask = coeff < 1
                gradient[mask] *= coeff[mask, None]

            # Correct the KL divergence w.r.t. the exaggeration if needed
            if should_eval_error and exaggeration != 1:
                error = error / exaggeration - np.log(exaggeration)

            if should_call_callback:
                state = IterationState(
                    iteration=iteration + 1,
                    exaggeration=exaggeration,
                    embedding=embedding.copy(),
                    gradient=gradient_snapshot,
                    error=error,
                    dof=dof_,
                    dof_grad=dof_grad,
                )
                # Continue only if all the callbacks say so
                should_stop = any(bool(c(state)) for c in callbacks)
                if should_stop:
                    # Make sure to un-exaggerate P so it's not corrupted in future runs
                    if exaggeration != 1:
                        P /= exaggeration
                    raise OptimizationInterrupt(error=error, final_embedding=embedding)

            # === DOF UPDATE (delta-bar-delta) ===
            if dof == "auto":
                dof_grad_direction_flipped = np.sign(self.dof_update) != np.sign(
                    dof_grad
                )
                if dof_grad_direction_flipped:
                    self.dof_gain += 0.2
                else:
                    self.dof_gain = self.dof_gain * 0.8 + min_gain

                # Normalize learning rate by n_samples to get appropriate scale for DoF
                n_samples = embedding.shape[0]
                dof_lr = (
                    learning_rate / n_samples
                )  # <<<< This is just 1 with default settings

                self.dof_update = (
                    momentum * self.dof_update - dof_lr * self.dof_gain * dof_grad
                )
                dof_ += self.dof_update
            # === END DOF UPDATE ===

            # Update the embedding using the gradient
            grad_direction_flipped = np.sign(self.update) != np.sign(gradient)
            grad_direction_same = np.invert(grad_direction_flipped)
            self.gains[grad_direction_flipped] += 0.2
            self.gains[grad_direction_same] = (
                self.gains[grad_direction_same] * 0.8 + min_gain
            )
            gradient = gradient.view(np.ndarray)
            self.update = momentum * self.update - learning_rate * self.gains * gradient

            # Clip the update sizes
            if max_step_norm is not None:
                update_norms = np.linalg.norm(self.update, axis=1, keepdims=True)
                mask = update_norms.squeeze() > max_step_norm
                self.update[mask] /= update_norms[mask]
                self.update[mask] *= max_step_norm

            embedding += self.update

            # Zero-mean the embedding only if we're not adding new data points,
            # otherwise this will reset point positions
            if reference_embedding is None:
                embedding -= np.mean(embedding, axis=0)

            # Limit any new points within the circle defined by the interpolation grid
            if should_limit_range:
                if embedding.shape[1] == 1:
                    mask = (embedding < lower_limit) | (embedding > upper_limit)
                    np.clip(embedding, lower_limit, upper_limit, out=embedding)
                elif embedding.shape[1] == 2:
                    r_limit = max(abs(lower_limit), abs(upper_limit))
                    embedding, mask = utils.clip_point_to_disc(
                        embedding, r_limit, inplace=True
                    )

                # Zero out the momentum terms for the points that hit the boundary
                self.gains[mask] = 0

            if verbose and (iteration + 1) % 50 == 0:
                stop_time = time()
                print(
                    "Iteration %4d, KL divergence %6.4f, 50 iterations in %.4f sec"
                    % (iteration + 1, error, stop_time - start_time)
                )
                start_time = time()

        timer.__exit__()

        # Make sure to un-exaggerate P so it's not corrupted in future runs
        if exaggeration != 1:
            P /= exaggeration

        # The error from the loop is the one for the previous, non-updated
        # embedding. We need to return the error for the actual final embedding, so
        # compute that at the end before returning
        result = objective_function(
            embedding,
            P,
            dof=dof_,
            bh_params=bh_params,
            fft_params=fft_params,
            reference_embedding=reference_embedding,
            n_jobs=n_jobs,
            should_eval_error=True,
            compute_dof_grad=False,
        )

        return result.error, embedding
