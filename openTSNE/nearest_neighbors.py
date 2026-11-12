import logging
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import neighbors
from sklearn.utils import check_random_state

from openTSNE import utils

log = logging.getLogger(__name__)


def _exclude_self(indices, distances):
    """Remove each point from its own list of ``k + 1`` nearest neighbors.

    Row ``i`` of ``indices`` and ``distances`` holds the neighbors of point
    ``i``. The point itself need not be the first result: duplicate points
    are at distance zero, and under the inner product a point need not be its
    own nearest neighbor. Rows that do not contain the point itself drop their
    furthest result.

    Returns
    -------
    indices: np.ndarray
    distances: np.ndarray
        Both of shape ``(n_samples, k)``.

    """
    n_samples, k = indices.shape[0], indices.shape[1] - 1
    is_self = indices == np.arange(n_samples)[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    keep = ~is_self
    return indices[keep].reshape(n_samples, k), distances[keep].reshape(n_samples, k)


class KNNIndex:
    VALID_METRICS = []
    supports_callable = False

    def __init__(
        self,
        data,
        k,
        metric="euclidean",
        metric_params=None,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_kwargs=None,
    ):
        self.data = data
        self.n_samples = data.shape[0]
        self.k = k
        self.metric = self.check_metric(metric)
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.knn_kwargs = knn_kwargs

        self.index = None

    def build(self):
        """Build the nearest neighbor index on the training data.

        Builds an index on the training data and computes the nearest neighbors
        on the training data.

        Returns
        -------
        indices: np.ndarray
        distances: np.ndarray

        """

    def query(self, query, k):
        """Query the index with new points.

        Finds k nearest neighbors from the training data to each row of the
        query data.

        Parameters
        ----------
        query: array_like
        k: int

        Returns
        -------
        indices: np.ndarray
        distances: np.ndarray

        """

    def check_metric(self, metric):
        """Check that the metric is supported by the KNNIndex instance."""
        if callable(metric):
            if not self.supports_callable:
                raise ValueError(
                    f"`{self.__class__.__name__}` does not support callable "
                    f"metrics. Please choose one of the supported metrics: "
                    f"{', '.join(self.VALID_METRICS)} or use one of the KNN "
                    f"methods that supports callable metrics: `exact` and "
                    f"`pynndescent`. You may first need to install pynndescent "
                    f"(pip install pynndescent)."
                )
        elif metric not in self.VALID_METRICS:
            raise ValueError(
                f"`{self.__class__.__name__}` does not support the `{metric}` "
                f"metric. Please choose one of the supported metrics: "
                f"{', '.join(self.VALID_METRICS)}."
            )

        return metric


class Sklearn(KNNIndex):
    VALID_METRICS = [
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "dice",
        "euclidean",
        "hamming",
        "haversine",
        "infinity",
        "jaccard",
        "kulsinski",
        "l1",
        "l2",
        "mahalanobis",
        "manhattan",
        "matching",
        "minkowski",
        "p",
        "pyfunc",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "wminkowski",
    ] + ["cosine"]  # our own workaround implementation
    supports_callable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__data = None

    def build(self):
        data, k = self.data, self.k

        timer = utils.Timer(
            f"Finding {k} nearest neighbors using exact search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        if self.metric == "cosine":
            # The nearest neighbor ranking for cosine distance is the same as
            # for euclidean distance on normalized data
            effective_metric = "euclidean"
            effective_data = data.copy()
            effective_data = (
                effective_data / np.linalg.norm(effective_data, axis=1)[:, None]
            )
            # In order to properly compute cosine distances when querying the
            # index, we need to store the original data
            self.__data = data
        else:
            effective_metric = self.metric
            effective_data = data

        knn_kwargs_ = dict(algorithm="auto")
        if self.knn_kwargs is not None:
            knn_kwargs_.update(self.knn_kwargs)

        self.index = neighbors.NearestNeighbors(
            metric=effective_metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
            **knn_kwargs_,
        )
        self.index.fit(effective_data)

        # Return the nearest neighbors in the training set
        distances, indices = self.index.kneighbors(n_neighbors=k)

        # If using cosine distance, the computed distances will be wrong and
        # need to be recomputed
        if self.metric == "cosine":
            distances = np.vstack(
                [
                    cdist(np.atleast_2d(x), data[idx], metric="cosine")
                    for x, idx in zip(data, indices)
                ]
            )

        timer.__exit__()

        return indices, distances

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using exact search...",
            self.verbose,
        )
        timer.__enter__()

        # The nearest neighbor ranking for cosine distance is the same as for
        # euclidean distance on normalized data
        if self.metric == "cosine":
            effective_data = query.copy()
            effective_data = (
                effective_data / np.linalg.norm(effective_data, axis=1)[:, None]
            )
        else:
            effective_data = query

        distances, indices = self.index.kneighbors(effective_data, n_neighbors=k)

        # If using cosine distance, the computed distances will be wrong and
        # need to be recomputed
        if self.metric == "cosine":
            if self.__data is None:
                raise RuntimeError(
                    "The original data was unavailable when querying cosine "
                    "distance. Did you change the distance metric after "
                    "building the index? Please rebuild the index using cosine "
                    "similarity."
                )
            distances = np.vstack(
                [
                    cdist(np.atleast_2d(x), self.__data[idx], metric="cosine")
                    for x, idx in zip(query, indices)
                ]
            )

        timer.__exit__()

        return indices, distances


class Annoy(KNNIndex):
    VALID_METRICS = [
        "cosine",
        "euclidean",
        "manhattan",
        "hamming",
        "dot",
        "l1",
        "l2",
        "taxicab",
    ]

    def build(self):
        data, k = self.data, self.k

        timer = utils.Timer(
            f"Finding {k} nearest neighbors using Annoy approximate search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        from openTSNE.dependencies.annoy import AnnoyIndex

        N = data.shape[0]

        annoy_metric = self.metric
        annoy_aliases = {
            "cosine": "angular",
            "l1": "manhattan",
            "l2": "euclidean",
            "taxicab": "manhattan",
        }
        if annoy_metric in annoy_aliases:
            annoy_metric = annoy_aliases[annoy_metric]

        self.index = AnnoyIndex(data.shape[1], annoy_metric)

        random_state = check_random_state(self.random_state)
        self.index.set_seed(random_state.randint(np.iinfo(np.int32).max))

        for i in range(N):
            self.index.add_item(i, data[i])

        # Number of trees. FIt-SNE uses 50 by default.
        knn_kwargs_ = {"n_trees": 50}
        if self.knn_kwargs is not None:
            knn_kwargs_.update(self.knn_kwargs)

        self.index.build(n_jobs=self.n_jobs, **knn_kwargs_)

        # Return the nearest neighbors in the training set, which include the
        # query points themselves
        distances = np.zeros((N, k + 1))
        indices = np.zeros((N, k + 1)).astype(int)

        def getnns(i):
            indices[i], distances[i] = self.index.get_nns_by_item(
                i, k + 1, include_distances=True
            )

        if self.n_jobs == 1:
            for i in range(N):
                getnns(i)
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=self.n_jobs, require="sharedmem")(
                delayed(getnns)(i) for i in range(N)
            )

        timer.__exit__()

        return _exclude_self(indices, self._convert_distances(distances))

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using Annoy "
            f"approximate search...",
            self.verbose,
        )
        timer.__enter__()

        N = query.shape[0]
        distances = np.zeros((N, k))
        indices = np.zeros((N, k)).astype(int)

        def getnns(i):
            indices[i], distances[i] = self.index.get_nns_by_vector(
                query[i], k, include_distances=True
            )

        if self.n_jobs == 1:
            for i in range(N):
                getnns(i)
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=self.n_jobs, require="sharedmem")(
                delayed(getnns)(i) for i in range(N)
            )

        timer.__exit__()

        return indices, self._convert_distances(distances)

    def _convert_distances(self, distances):
        """Convert distances reported by Annoy to the scale of the metric."""
        if self.metric == "cosine":
            # Annoy's angular distance is the euclidean distance between the
            # normalized vectors, sqrt(2 - 2 cos)
            return distances ** 2 / 2
        if self.metric == "dot":
            # Annoy ranks by the largest inner product and reports the inner
            # product itself, a similarity. One minus the inner product is a
            # distance for normalized data, matching HNSW and pynndescent.
            return 1 - distances
        return distances

    def __getstate__(self):
        import tempfile
        import base64
        from os import path

        d = dict(self.__dict__)
        # If the index is not None, we want to save the encoded index
        if self.index is not None:
            with tempfile.TemporaryDirectory() as dirname:
                self.index.save(path.join(dirname, "tmp.ann"))

                with open(path.join(dirname, "tmp.ann"), "rb") as f:
                    b64_index = base64.b64encode(f.read())

            d["b64_index"] = b64_index
            del d["index"]

        return d

    def __setstate__(self, state):
        import tempfile
        import base64
        from os import path

        from openTSNE.dependencies.annoy import AnnoyIndex

        # If a base64 index is given, we have to load the index
        if "b64_index" in state:
            assert "index" not in state
            b64_index = state["b64_index"]
            del state["b64_index"]

            annoy_metric = state["metric"]
            annoy_aliases = {
                "cosine": "angular",
                "l1": "manhattan",
                "l2": "euclidean",
                "taxicab": "manhattan",
            }
            if annoy_metric in annoy_aliases:
                annoy_metric = annoy_aliases[annoy_metric]

            self.index = AnnoyIndex(state["data"].shape[1], annoy_metric)
            with tempfile.TemporaryDirectory() as dirname:
                with open(path.join(dirname, "tmp.ann"), "wb") as f:
                    f.write(base64.b64decode(b64_index))
                self.index.load(path.join(dirname, "tmp.ann"))

        self.__dict__.update(state)


class NNDescent(KNNIndex):
    VALID_METRICS = [
        "euclidean",
        "l2",
        "sqeuclidean",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "linfinity",
        "linfty",
        "linf",
        "minkowski",
        "seuclidean",
        "standardised_euclidean",
        "wminkowski",
        "weighted_minkowski",
        "mahalanobis",
        "canberra",
        "cosine",
        "dot",
        "correlation",
        "haversine",
        "braycurtis",
        "spearmanr",
        "tsss",
        "true_angular",
        "hellinger",
        "kantorovich",
        "wasserstein",
        "wasserstein_1d",
        "wasserstein-1d",
        "kantorovich-1d",
        "kantorovich_1d",
        "circular_kantorovich",
        "circular_wasserstein",
        "sinkhorn",
        "jensen-shannon",
        "jensen_shannon",
        "symmetric-kl",
        "symmetric_kl",
        "symmetric_kullback_liebler",
        "hamming",
        "jaccard",
        "dice",
        "matching",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalsneath",
        "sokalmichener",
        "yule",
        "bit_hamming",
        "bit_jaccard",
    ]
    supports_callable = True

    def __init__(self, *args, **kwargs):
        try:
            import pynndescent  # pylint: disable=unused-import,unused-variable
        except ImportError:
            raise ImportError(
                "Please install pynndescent: `conda install -c conda-forge "
                "pynndescent` or `pip install pynndescent`."
            )
        super().__init__(*args, **kwargs)

    def check_metric(self, metric):
        import pynndescent

        if not np.array_equal(
            list(pynndescent.distances.named_distances), self.VALID_METRICS
        ):
            warnings.warn(
                "`pynndescent` has recently changed which distance metrics are supported, "
                "and `openTSNE.nearest_neighbors` has not been updated. Please notify the "
                "developers of this change."
            )

        if callable(metric):
            from numba.core.registry import CPUDispatcher

            if not isinstance(metric, CPUDispatcher):
                warnings.warn(
                    f"`pynndescent` requires callable metrics to be "
                    f"compiled with `numba`, but `{metric.__name__}` is not compiled. "
                    f"`openTSNE.nearest_neighbors.NNDescent` "
                    f"will attempt to compile the function. "
                    f"If this results in an error, then the function may not be "
                    f"compatible with `numba.njit` and should be rewritten. "
                    f"Otherwise, set `neighbors`='exact' to use `scikit-learn` "
                    f"for calculating nearest neighbors."
                )
                from numba import njit

                metric = njit(fastmath=True)(metric)

        return super().check_metric(metric)

    def build(self):
        data, k = self.data, self.k

        timer = utils.Timer(
            f"Finding {k} nearest neighbors using NN descent approximate search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        # Numba takes a while to load up, so there's little point in loading it
        # unless we're actually going to use it
        import pynndescent

        # Will use query() only for k>15
        if k <= 15:
            n_neighbors_build = k + 1
        else:
            n_neighbors_build = 15

        knn_kwargs_ = {
            "n_trees": 5 + int(round((self.data.shape[0]) ** 0.5 / 20)),
            "n_iters": max(5, int(round(np.log2(self.data.shape[0])))),
            "max_candidates": 60,
        }
        if self.knn_kwargs is not None:
            knn_kwargs_.update(self.knn_kwargs)

        self.index = pynndescent.NNDescent(
            data,
            n_neighbors=n_neighbors_build,
            metric=self.metric,
            metric_kwds=self.metric_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose > 1,
            **knn_kwargs_,
        )

        # -1 in indices means that pynndescent failed
        indices, distances = self.index.neighbor_graph
        mask = np.sum(indices == -1, axis=1) > 0

        if k > n_neighbors_build:
            indices, distances = self.index.query(data, k=k + 1)

        # As a workaround, we let the failed points group together
        if np.sum(mask) > 0:
            if self.verbose:
                opt = np.get_printoptions()
                np.set_printoptions(threshold=np.inf)
                warnings.warn(
                    f"`pynndescent` failed to find neighbors for some of the points. "
                    f"As a workaround, openTSNE considers all such points similar to "
                    f"each other, so they will likely form a cluster in the embedding."
                    f"The indices of the failed points are:\n{np.where(mask)[0]}"
                )
                np.set_printoptions(**opt)
            else:
                warnings.warn(
                    f"`pynndescent` failed to find neighbors for some of the points. "
                    f"As a workaround, openTSNE considers all such points similar to "
                    f"each other, so they will likely form a cluster in the embedding. "
                    f"Run with verbose=True, to see indices of the failed points."
                )
            distances[mask] = 1
            rs = check_random_state(self.random_state)
            fake_indices = rs.choice(
                np.sum(mask), size=np.sum(mask) * indices.shape[1], replace=True
            )
            fake_indices = np.where(mask)[0][fake_indices]
            indices[mask] = np.reshape(fake_indices, (np.sum(mask), indices.shape[1]))

        timer.__exit__()

        # pynndescent returns float32 distances; the other indices return float64
        return indices[:, 1:], distances[:, 1:].astype(np.float64)

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using NN Descent "
            f"approxmimate search...",
            self.verbose,
        )
        timer.__enter__()

        indices, distances = self.index.query(query, k=k)

        timer.__exit__()

        return indices, distances.astype(np.float64)


class HNSW(KNNIndex):
    """Approximate nearest neighbor search using hierarchical navigable small
    world graphs, as implemented in hnswlib.

    Distances are returned on the scale of the metric: ``euclidean`` and
    ``l2`` give euclidean distances, ``cosine`` gives one minus the cosine
    similarity, and ``ip`` and ``dot`` give one minus the inner product. The
    inner-product distances are negative whenever the inner product exceeds
    one, so they are only meaningful as t-SNE input for normalized data.

    The ``M`` and ``ef_construction`` entries of ``knn_kwargs`` are passed to
    ``hnswlib.Index.init_index``. ``ef`` sets the size of the candidate list
    searched for each query point, trading speed for recall; it defaults to
    ``max(2 * k, 100)``. Building the index with more than one thread is not
    deterministic, so results are reproducible with a fixed ``random_state``
    only when ``n_jobs=1``.

    """
    VALID_METRICS = [
        "cosine",
        "euclidean",
        "dot",
        "l2",
        "ip",
    ]
    HNSW_SPACES = {
        "cosine": "cosine",
        "dot": "ip",
        "euclidean": "l2",
        "ip": "ip",
        "l2": "l2",
    }

    def __init__(self, *args, **kwargs):
        try:
            from hnswlib import Index  # pylint: disable=unused-import,unused-variable
        except ImportError:
            raise ImportError(
                "Please install hnswlib: `conda install -c conda-forge "
                "hnswlib` or `pip install hnswlib`."
            )
        super().__init__(*args, **kwargs)

    def build(self):
        data, k = self.data, self.k

        timer = utils.Timer(
            f"Finding {k} nearest neighbors using HNSWlib approximate search using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        from hnswlib import Index

        hnsw_space = self.HNSW_SPACES[self.metric]

        random_state = check_random_state(self.random_state)
        random_seed = random_state.randint(np.iinfo(np.int32).max)

        self.index = Index(space=hnsw_space, dim=data.shape[1])

        knn_kwargs_ = dict(ef_construction=200, M=16)
        if self.knn_kwargs is not None:
            knn_kwargs_.update(self.knn_kwargs)
        # `ef` is a search-time parameter, set in `_set_ef`
        knn_kwargs_.pop("ef", None)

        # Initialize HNSW Index
        self.index.init_index(
            max_elements=data.shape[0],
            random_seed=random_seed,
            **knn_kwargs_,
        )

        num_threads = self._num_threads()
        if self.random_state is not None and num_threads > 1:
            warnings.warn(
                "Building an HNSW index with multiple threads is not "
                "deterministic, so the nearest neighbors may differ between "
                "runs even with a fixed `random_state`. Set `n_jobs=1` for "
                "reproducible results."
            )

        # Build index tree from data
        self.index.add_items(data, num_threads=num_threads)

        self._set_ef(k)

        # Query for kNN
        indices, distances = self.index.knn_query(data, k=k + 1, num_threads=num_threads)

        # Stop timer
        timer.__exit__()

        return _exclude_self(*self._convert_output(indices, distances))

    def query(self, query, k):
        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using HNSWlib "
            f"approximate search...",
            self.verbose,
        )
        timer.__enter__()

        self._set_ef(k)

        # Query for kNN
        indices, distances = self.index.knn_query(
            query, k=k, num_threads=self._num_threads()
        )

        # Stop timer
        timer.__exit__()

        # return indices and distances
        return self._convert_output(indices, distances)

    def _num_threads(self):
        """Resolve `n_jobs` following the scikit-learn convention, ``-1``
        meaning all processors, ``-2`` all but one, etc. hnswlib itself treats
        every non-positive value as all processors."""
        from joblib import effective_n_jobs

        return effective_n_jobs(self.n_jobs)

    def _set_ef(self, k):
        """Set the size of the candidate list searched for each query point.

        Larger values give higher recall at the cost of slower queries. The
        default, ``max(2 * k, 100)``, can be overridden through the ``ef`` key
        of ``knn_kwargs``. The lower bound keeps recall high for small ``k``,
        where a candidate list of ``2 * k`` misses many true neighbors.
        """
        ef = None
        if self.knn_kwargs is not None:
            ef = self.knn_kwargs.get("ef")
        if ef is None:
            ef = max(2 * k, 100)
        self.index.set_ef(min(ef, self.index.get_current_count()))

    def _convert_output(self, indices, distances):
        """Convert hnswlib results to the conventions of the other indices:
        int64 indices and float64 distances on the scale of the metric."""
        indices = indices.astype(np.int64)
        distances = distances.astype(np.float64)
        # The hnswlib `l2` space returns squared euclidean distances
        if self.HNSW_SPACES[self.metric] == "l2":
            np.sqrt(distances, out=distances)
        return indices, distances

    def __setstate__(self, state):
        # `hnswlib.Index` pickles natively, so `index` is part of the state.
        # Pickles written by openTSNE<=1.0.4 instead store the index as a
        # base64-encoded hnswlib index file under `b64_index`; loading these
        # is deprecated.
        if "b64_index" in state:
            warnings.warn(
                "This HNSW index was pickled with openTSNE<=1.0.4. Loading this "
                "format is deprecated and will be removed in a future version. "
                "Please re-save the object to store it in the current format.",
                category=FutureWarning,
            )

            import tempfile
            import base64
            from os import path

            from hnswlib import Index

            assert "index" not in state
            state = dict(state)
            b64_index = state.pop("b64_index")

            index = Index(
                space=self.HNSW_SPACES[state["metric"]],
                dim=state["data"].shape[1],
            )
            with tempfile.TemporaryDirectory() as dirname:
                with open(path.join(dirname, "tmp.bin"), "wb") as f:
                    f.write(base64.b64decode(b64_index))
                index.load_index(path.join(dirname, "tmp.bin"))
            state["index"] = index

        self.__dict__.update(state)


class MLXVS(KNNIndex):
    """Nearest neighbor search on the GPU of Apple silicon Macs, using mlxvs.

    mlxvs (https://github.com/mikepapadim/mlxvs) runs on Apple's MLX and needs
    macOS 14 or later on Apple silicon. Up to ``exact_below`` points (the
    ``exact_below`` entry of ``knn_kwargs``, 50,000 by default) the neighbors
    are exact: a matrix multiply and a top-k on the GPU, the same on every run.
    Larger data sets get approximate neighbors from NN-Descent k-nearest
    neighbor graphs, built in overlapping k-means clusters; these can differ
    slightly between runs, even with a fixed ``random_state``. ``query`` then
    searches a CAGRA graph index of the training data (``itopk_size`` entry of
    ``knn_kwargs``: the search list length, 64 by default, longer is more
    accurate). Distances are recomputed in double precision for the returned
    neighbors: euclidean distances for ``euclidean`` and ``l2``, one minus the
    cosine similarity for ``cosine``.

    """
    VALID_METRICS = [
        "cosine",
        "euclidean",
        "l2",
    ]

    def __init__(self, *args, **kwargs):
        try:
            import mlxvs  # pylint: disable=unused-import,unused-variable
        except ImportError:
            raise ImportError(
                "Please install mlxvs (macOS on Apple silicon only): "
                "`pip install mlxvs`."
            )
        super().__init__(*args, **kwargs)

    def _setting(self, name, default):
        if self.knn_kwargs is not None and name in self.knn_kwargs:
            return self.knn_kwargs[name]
        return default

    def _mlxvs_metric(self):
        return "cosine" if self.metric == "cosine" else "l2"

    def _exact(self):
        return self.n_samples < self._setting("exact_below", 50_000)

    def build(self):
        import mlxvs

        data, k = np.ascontiguousarray(self.data, dtype=np.float32), self.k

        timer = utils.Timer(
            f"Finding {k} nearest neighbors using mlxvs on the GPU "
            f"({'exact' if self._exact() else 'approximate'} search) using "
            f"{self.metric} distance...",
            verbose=self.verbose,
        )
        timer.__enter__()

        if self._exact():
            indices, _ = mlxvs.brute_force_knn(
                data, data, min(k + 1, self.n_samples), self._mlxvs_metric()
            )
            indices = np.asarray(indices).astype(np.int64)
            indices = _exclude_self(indices, indices)[0]
        else:
            if self.random_state is not None:
                warnings.warn(
                    "mlxvs builds approximate nearest neighbor graphs of more "
                    f"than {self._setting('exact_below', 50_000)} points with "
                    "NN-Descent, which is not deterministic, so the nearest "
                    "neighbors may differ slightly between runs even with a "
                    "fixed `random_state`."
                )
            indices, _ = mlxvs.all_neighbors(data, k, self._mlxvs_metric())
            indices = np.asarray(indices).astype(np.int64)

        timer.__exit__()

        return indices, self._distances(self.data, indices)

    def query(self, query, k):
        import mlxvs

        timer = utils.Timer(
            f"Finding {k} nearest neighbors in existing embedding using mlxvs "
            f"on the GPU...",
            self.verbose,
        )
        timer.__enter__()

        query32 = np.ascontiguousarray(query, dtype=np.float32)
        k_ = min(k, self.n_samples)
        if self._exact():
            indices, _ = mlxvs.brute_force_knn(
                np.ascontiguousarray(self.data, dtype=np.float32),
                query32,
                k_,
                self._mlxvs_metric(),
            )
        else:
            if self.index is None:
                self.index = mlxvs.CagraIndex.build(
                    np.ascontiguousarray(self.data, dtype=np.float32),
                    32,
                    self._mlxvs_metric(),
                )
            indices, _ = self.index.search(
                query32, k_, itopk_size=max(self._setting("itopk_size", 64), k_)
            )
        indices = np.asarray(indices).astype(np.int64)

        timer.__exit__()

        return indices, self._distances(query, indices)

    def _distances(self, points, indices):
        """The distances from each point to its neighbors, in double precision,
        computed in blocks of points so that memory stays bounded."""
        points = np.asarray(points)
        data = np.asarray(self.data)
        n, k = indices.shape
        distances = np.empty((n, k), dtype=np.float64)
        block = max(1, (64 << 20) // (8 * max(1, k * data.shape[1])))
        for start in range(0, n, block):
            rows = slice(start, min(n, start + block))
            p = points[rows].astype(np.float64)
            neighbors = data[indices[rows]].astype(np.float64)
            if self.metric == "cosine":
                dots = np.einsum("ij,ikj->ik", p, neighbors)
                norms = np.linalg.norm(p, axis=1)[:, None] * np.linalg.norm(
                    neighbors, axis=2
                )
                distances[rows] = 1 - dots / np.maximum(
                    norms, np.finfo(np.float64).tiny
                )
            else:
                distances[rows] = np.sqrt(
                    ((neighbors - p[:, None, :]) ** 2).sum(axis=2)
                )
        return distances

    def __getstate__(self):
        # the GPU index is rebuilt on demand
        state = dict(self.__dict__)
        state["index"] = None
        return state


class PrecomputedDistanceMatrix(KNNIndex):
    """Use a precomputed distance matrix to construct the KNNG.

    Parameters
    ----------
    distance_matrix: np.ndarray
        A square, symmetric, and contain only poistive values.

    """
    def __init__(self, distance_matrix, k):
        nn = neighbors.NearestNeighbors(metric="precomputed")
        nn.fit(distance_matrix)
        self.distances, self.indices = nn.kneighbors(n_neighbors=k)
        self.n_samples = distance_matrix.shape[0]
        self.k = k

    def build(self):
        return self.indices, self.distances

    def query(self, query, k):
        """Use a precomputed distance matrix to determine the KNNG for the
        transformed samples.

        Parameters
        ----------
        query: array_like
            An M x N distance matrix where M is the number of query points and
            N is the number of samples in the existing embedding.
        k: int

        Returns
        -------
        indices: np.ndarray
        distances: np.ndarray

        """
        indices = np.argsort(query, axis=1)[:, :k]
        distances = np.take_along_axis(query, indices, axis=1)
        return indices, distances


class PrecomputedNeighbors(KNNIndex):
    """Use a precomputed distance matrix to construct the KNNG.

    Parameters
    ----------
    neighbors: np.ndarray
        A N x K matrix containing the indices of point i's k nearest neighbors.

    distances: np.ndarray
        A N x K matrix containing the distances to from data point i to its k
        nearest neighbors.

    """
    def __init__(self, neighbors, distances):
        self.distances, self.indices = distances, neighbors
        self.n_samples = neighbors.shape[0]
        self.k = neighbors.shape[1]

    def build(self):
        return self.indices, self.distances

    def query(self, *args, **kwargs):
        raise RuntimeError("Precomputed distance matrices cannot be queried")
