import unittest
from functools import partial

from scipy.spatial.distance import pdist, squareform

import openTSNE
import openTSNE.affinity
import openTSNE.initialization
import numpy as np
from openTSNE.callbacks import ErrorApproximations, VerifyExaggerationError
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import SpectralEmbedding

TSNE = partial(openTSNE.TSNE, neighbors="exact", negative_gradient_method="bh")


class TestTSNECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE(early_exaggeration_iter=20, n_iter=100)
        # Set up two modalities, if we want to viually inspect test results
        random_state = np.random.RandomState(0)
        cls.x = np.vstack(
            (random_state.normal(+1, 1, (100, 4)), random_state.normal(-1, 1, (100, 4)))
        )
        cls.x_test = random_state.normal(0, 1, (25, 4))
        cls.iris = datasets.load_iris()

    def test_basic_flow(self):
        """Verify that the basic flow does not crash."""
        embedding = self.tsne.fit(self.x)
        self.assertFalse(np.any(np.isnan(embedding)))

        partial_embedding = embedding.transform(self.x_test, n_iter=20)
        self.assertFalse(np.any(np.isnan(partial_embedding)))

    def test_advanced_flow(self):
        """Verify that the advanced flow does not crash."""
        embedding = self.tsne.prepare_initial(self.x)
        embedding = embedding.optimize(20, exaggeration=12)
        embedding = embedding.optimize(20)  # type: openTSNE.TSNEEmbedding
        self.assertFalse(np.any(np.isnan(embedding)))

        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding = partial_embedding.optimize(20, exaggeration=2)
        partial_embedding = partial_embedding.optimize(20)
        self.assertFalse(np.any(np.isnan(partial_embedding)))

    def test_error_exaggeration_correction(self):
        embedding = self.tsne.prepare_initial(self.x)

        # The callback raises if the KL divergence does not match the true one
        embedding.optimize(
            50,
            exaggeration=5,
            callbacks=[VerifyExaggerationError(embedding)],
            callbacks_every_iters=1,
            inplace=True,
        )

    def test_error_exaggeration_correction_with_learned_dof(self):
        for method in ("bh", "fft", "auto"):
            with self.subTest(method=method):
                tsne = openTSNE.TSNE(
                    neighbors="exact", negative_gradient_method=method, dof="auto"
                )
                embedding = tsne.prepare_initial(self.x)
                embedding.optimize(
                    10,
                    exaggeration=5,
                    callbacks=[VerifyExaggerationError(embedding)],
                    callbacks_every_iters=1,
                    inplace=True,
                )
                self.assertNotEqual(embedding.dof_, 1)

    def test_error_approximations_use_the_iteration_dof(self):
        embedding = TSNE(dof=5, early_exaggeration_iter=0, n_iter=20).fit(self.x)
        approximations = ErrorApproximations(embedding.affinities.P)
        states = []
        embedding.optimize(
            3,
            callbacks=[approximations, states.append],
            callbacks_every_iters=1,
            inplace=True,
        )
        np.testing.assert_allclose(
            approximations.bh_errors, [s.error for s in states], rtol=1e-10
        )

    def test_iris(self):
        x, y = self.iris.data, self.iris.target

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        tsne = TSNE(perplexity=30, initialization="random", random_state=0)

        # Prepare a random initialization
        embedding = tsne.prepare_initial(x)

        # KNN should do poorly on a random initialization
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertLess(accuracy_score(predictions, y), 0.5)

        # Optimize the embedding for a small number of steps so tests run fast
        embedding.optimize(250, inplace=True)

        # Similar points should be grouped together, therefore KNN should do well
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertGreater(accuracy_score(predictions, y), 0.95)

    def test_iris_with_precomputed_distance_matrices(self):
        x, y = self.iris.data, self.iris.target

        distances = squareform(pdist(x))

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        tsne = TSNE(
            perplexity=30, initialization="random", random_state=0, metric="precomputed"
        )

        # Prepare a random initialization
        embedding = tsne.prepare_initial(distances)

        # KNN should do poorly on a random initialization
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertLess(accuracy_score(predictions, y), 0.5)

        # Optimize the embedding for a small number of steps so tests run fast
        embedding.optimize(250, inplace=True)

        # Similar points should be grouped together, therefore KNN should do well
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertGreater(accuracy_score(predictions, y), 0.95)

    def test_iris_bh_transform_equivalency_with_one_by_one(self):
        """Compare one by one embedding vs all at once using BH gradients."""
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.1, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            early_exaggeration_iter=0,
            n_iter=50,
            neighbors="exact",
            negative_gradient_method="bh",
        ).fit(x_train)

        params = dict(n_iter=100, perplexity=5)
        # Build up an embedding by adding points one by one
        new_embedding_1 = np.vstack(
            [embedding.transform(np.atleast_2d(point), **params) for point in x_test]
        )
        # Add new points altogether
        new_embedding_2 = embedding.transform(x_test, **params)

        # Verify that the embedding has actually been optimized
        self.assertRaises(
            AssertionError,
            np.testing.assert_almost_equal,
            embedding.prepare_partial(x_test, perplexity=params["perplexity"]),
            new_embedding_1,
        )
        # Check that both methods produced the same embedding
        np.testing.assert_almost_equal(new_embedding_1, new_embedding_2)

    def test_iris_fft_transform_equivalency_with_one_by_one(self):
        """Compare one by one embedding vs all at once using FFT gradients.

        Note that this won't return the exact same embedding both times because
        the grid placed over the embedding will differ when placing points one
        by one vs. when placing them at once. The min/max coords will differ,
        thus changing the overall approximation. They should be quite similar
        though.

        """
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.1, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            early_exaggeration_iter=0,
            n_iter=50,
            neighbors="exact",
            negative_gradient_method="fft",
        ).fit(x_train)

        # Changing the gradients using clipping changes how the points move
        # sufficiently so that the interpolation grid is shifted. This test is
        # more reliable when we don't do gradient clipping and reduce the
        # learning rate. We increase the number of iterations so that the points
        # have time to move around
        params = dict(perplexity=5)
        # Build up an embedding by adding points one by one
        new_embedding_1 = np.vstack(
            [embedding.transform(np.atleast_2d(point), **params) for point in x_test]
        )
        # Add new points altogether
        new_embedding_2 = embedding.transform(x_test, **params)

        # Verify that the embedding has actually been optimized
        self.assertRaises(
            AssertionError,
            np.testing.assert_almost_equal,
            embedding.prepare_partial(x_test, perplexity=params["perplexity"]),
            new_embedding_1,
        )

        # Check that both methods produced the same embedding
        np.testing.assert_almost_equal(new_embedding_1, new_embedding_2, decimal=2)

    def test_iris_bh_transform_correctness(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.iris.data, self.iris.target, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            neighbors="exact",
            negative_gradient_method="bh",
            early_exaggeration_iter=0,
            n_iter=50,
            random_state=0,
        ).fit(x_train)

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, y_train)

        new_embedding = embedding.transform(x_test, n_iter=100)
        predictions = knn.predict(new_embedding)
        self.assertGreater(accuracy_score(predictions, y_test), 0.95)

    def test_iris_fft_transform_correctness(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.iris.data, self.iris.target, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        embedding = openTSNE.TSNE(
            neighbors="exact",
            negative_gradient_method="fft",
            early_exaggeration_iter=0,
            n_iter=50,
            random_state=0,
        ).fit(x_train)

        # Evaluate t-SNE optimization using a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, y_train)

        new_embedding = embedding.transform(x_test, n_iter=100)
        predictions = knn.predict(new_embedding)
        self.assertGreater(accuracy_score(predictions, y_test), 0.95)

    def test_bh_transform_with_point_subsets_using_perplexity_nn(self):
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        init = openTSNE.initialization.pca(x_train)
        affinity = openTSNE.affinity.PerplexityBasedNN(x_train, method="exact")
        embedding = openTSNE.TSNEEmbedding(
            init, affinity, negative_gradient_method="bh", random_state=42
        )
        embedding.optimize(n_iter=50, inplace=True)

        # The test set contains 50 samples, so let's verify on half of those
        transform_params = dict(n_iter=0, early_exaggeration_iter=0)
        new_embedding_1 = embedding.transform(x_test, **transform_params)[:25]
        new_embedding_2 = embedding.transform(x_test[:25], **transform_params)

        np.testing.assert_equal(new_embedding_1, new_embedding_2)

    def test_fft_transform_with_point_subsets_using_perplexity_nn(self):
        x_train, x_test = train_test_split(
            self.iris.data, test_size=0.33, random_state=42
        )

        # Set up the initial embedding
        init = openTSNE.initialization.pca(x_train)
        affinity = openTSNE.affinity.PerplexityBasedNN(x_train, method="exact")
        embedding = openTSNE.TSNEEmbedding(
            init, affinity, negative_gradient_method="fft", random_state=42
        )
        embedding.optimize(n_iter=100, inplace=True)

        # The test set contains 50 samples, so let's verify on half of those
        transform_params = dict(n_iter=0, early_exaggeration_iter=0)
        new_embedding_1 = embedding.transform(x_test, **transform_params)[:25]
        new_embedding_2 = embedding.transform(x_test[:25], **transform_params)

        np.testing.assert_equal(new_embedding_1, new_embedding_2)

    def test_bh_with_n_components_gt_3(self):
        tsne = openTSNE.TSNE(
            n_components=4,
            negative_gradient_method="bh",
            neighbors="exact",
        )
        x = np.random.rand(100, 100)
        with self.assertWarns(FutureWarning):
        # with self.assertRaises(RuntimeError):
            tsne.fit(x)

    def test_fft_with_n_components_gt_2(self):
        tsne = openTSNE.TSNE(
            n_components=3,
            negative_gradient_method="fft",
            neighbors="exact",
        )
        x = np.random.rand(100, 100)
        with self.assertRaises(RuntimeError):
            tsne.fit(x)


class TestTSNECorrectnessUsingNonStandardDof(TestTSNECorrectness):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE(early_exaggeration_iter=20, n_iter=100, dof=0.8)
        # Set up two modalities, if we want to viually inspect test results
        random_state = np.random.RandomState(0)
        cls.x = np.vstack(
            (random_state.normal(+1, 1, (100, 4)), random_state.normal(-1, 1, (100, 4)))
        )
        cls.x_test = random_state.normal(0, 1, (25, 4))
        cls.iris = datasets.load_iris()


class TestTSNECorrectnessUsingPrecomputedDistanceMatrix(unittest.TestCase):
    def test_iris(self):
        rng = np.random.RandomState(0)

        x = datasets.load_iris().data
        x += rng.normal(0, 1e-3, x.shape)  # iris contains duplicate rows

        # We don't run for any iterations, because even though the distances are
        # the *same*, they are only the same up to float64 machine precision,
        # which is 1e-16. If we optimize this, the numeric errors will
        # eventually build up and result in slightly different embeddings
        # (visually indistinguishable).
        # See https://github.com/pavlin-policar/openTSNE/issues/247 to see this
        # problem in action. However, we can also check for correctness if the P
        # matrix is the same (up to machine precision), since this will lead to
        # the same embedding.
        distances = squareform(pdist(x))
        params = dict(
            early_exaggeration_iter=0,
            n_iter=0,
            initialization="random",
            random_state=0,
        )
        embedding1 = TSNE(metric="precomputed", **params).fit(distances)
        embedding2 = TSNE(metric="euclidean", **params).fit(x)

        np.testing.assert_almost_equal(
            embedding1.affinities.P.toarray(),
            embedding2.affinities.P.toarray(),
            decimal=16,
        )


class TestSpectralInitializationCorrectness(unittest.TestCase):
    def test_spectral_agreement_with_sklearn(self):
        # Generate some random data and stretch it, to give it some structure
        np.random.seed(42)
        x = np.random.randn(100, 20)
        x[:, 0] *= 5

        # Perform spectral embedding via sklearn and via openTSNE
        P = openTSNE.affinity.PerplexityBasedNN(x).P
        embedding1 = openTSNE.initialization.spectral(P, tol=0, add_jitter=False)
        embedding2 = SpectralEmbedding(affinity="precomputed").fit_transform(P)

        np.testing.assert_almost_equal(
            np.abs(np.corrcoef(embedding1[:, 0], embedding2[:, 0])[0, 1]), 1
        )
        np.testing.assert_almost_equal(
            np.abs(np.corrcoef(embedding1[:, 1], embedding2[:, 1])[0, 1]), 1
        )


class TestEarlyExaggerationCollapse(unittest.TestCase):
    """In some cases, the BH implementation was producing a collapsed embedding
    for all data points. For more information, see #233, #234."""
    def test_early_exaggeration_does_not_collapse(self):
        n_samples = [100, 150, 200]
        n_dims = [5, 10, 20]

        np.random.seed(42)
        for n in n_samples:
            for d in n_dims:
                x = np.random.randn(n, d)
                embedding = openTSNE.TSNE(random_state=42).fit(x)
                self.assertGreater(np.max(np.abs(embedding)), 1e-8)


class TestDataMatricesWithDuplicatedRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sklearn.preprocessing import KBinsDiscretizer

        # Load up contrived example where we have a large number of duplicated
        # rows. This is similar to the Titanic data set, which is problematic.
        np.random.seed(0)
        iris = datasets.load_iris()
        x, y = iris.data, iris.target

        discretizer = KBinsDiscretizer(n_bins=2, strategy="uniform")
        x = discretizer.fit_transform(x).toarray()

        idx = np.random.choice(x.shape[0], size=1000, replace=True)
        cls.x, cls.y = x[idx], y[idx]

    def test_works_without_error(self):
        openTSNE.TSNE(
            early_exaggeration=100, negative_gradient_method="bh", random_state=0
        ).fit(self.x)


class TestFFTGradientConcurrentThreads(unittest.TestCase):
    """The FFT gradient reuses buffers between calls. Embeddings optimized at
    the same time in different threads must still each get their own result."""

    def check_concurrent_calls_match_serial(self, n_dims, f, spans):
        from concurrent.futures import ThreadPoolExecutor

        random_state = np.random.RandomState(0)
        # Different spans give different grid sizes, so the threads also
        # disagree on the size of every buffer they ask for. Computing the dof
        # gradient adds a third kernel, and with it a third buffer
        cases = [
            (random_state.normal(0, span, (2000, n_dims)), dof, compute_dof_grad)
            for span in spans
            for dof in (1, 0.7)
            for compute_dof_grad in (False, True)
        ]

        def run(case):
            embedding, dof, compute_dof_grad = case
            gradient = np.zeros_like(embedding)
            if n_dims == 1:
                sum_q, dof_grad = f(
                    embedding.ravel(), gradient.ravel(), dof=dof,
                    compute_dof_grad=compute_dof_grad,
                )
            else:
                sum_q, dof_grad = f(
                    embedding, gradient, dof=dof, compute_dof_grad=compute_dof_grad
                )
            return gradient, sum_q, dof_grad

        expected = [run(case) for case in cases]
        # A dof gradient that is always zero would make its comparison vacuous
        for (_, _, compute_dof_grad), (_, _, dof_grad) in zip(cases, expected):
            self.assertEqual(dof_grad != 0, compute_dof_grad)

        with ThreadPoolExecutor(max_workers=len(cases)) as executor:
            for _ in range(5):
                results = list(executor.map(run, cases))
                for result, exp in zip(results, expected):
                    gradient, sum_q, dof_grad = result
                    exp_gradient, exp_sum_q, exp_dof_grad = exp
                    np.testing.assert_allclose(gradient, exp_gradient, rtol=1e-10)
                    np.testing.assert_allclose(sum_q, exp_sum_q, rtol=1e-10)
                    np.testing.assert_allclose(dof_grad, exp_dof_grad, rtol=1e-10)

    def test_1d(self):
        from openTSNE import _tsne

        self.check_concurrent_calls_match_serial(
            1, _tsne.estimate_negative_gradient_fft_1d, spans=(50, 500, 5000)
        )

    def test_2d(self):
        from openTSNE import _tsne

        self.check_concurrent_calls_match_serial(
            2, _tsne.estimate_negative_gradient_fft_2d, spans=(5, 20, 60)
        )
