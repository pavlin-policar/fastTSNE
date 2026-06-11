import inspect
import logging
import pickle
import unittest
from functools import wraps, partial
from typing import Callable, Any, Tuple, Optional
from unittest.mock import patch, MagicMock

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import openTSNE
from openTSNE import affinity
from openTSNE import initialization
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.nearest_neighbors import NNDescent
from openTSNE.tsne import (
    kl_divergence_bh, kl_divergence_fft, TSNEEmbedding, PartialTSNEEmbedding,
    GradientResult,
)
from openTSNE.utils import is_package_installed

np.random.seed(42)
affinity.log.setLevel(logging.ERROR)

TSNE = partial(openTSNE.TSNE, neighbors="exact", negative_gradient_method="bh")


def check_params(params: dict) -> Callable:
    """Run a series of parameterized tests to check tSNE parameter flow."""

    def _decorator(test_case: Callable) -> Callable:
        @wraps(test_case)
        def _wrapper(self):
            for param_name in params:
                for param_value in params[param_name]:
                    test_case(self, param_name, param_value)

        return _wrapper

    return _decorator


def check_call_contains_kwargs(
    call: Tuple,
    params: dict,
    param_mapping: Optional[dict] = None,
) -> None:
    """Check whether a `call` object was called with some params, but also some
    others we don't care about"""
    _param_mapping = {"negative_gradient_method": "objective_function",
                      "early_exaggeration_iter": "n_iter",
                      "late_exaggeration_iter": "n_iter",
                      "early_exaggeration": "exaggeration",
                      "late_exaggeration": "exaggeration",
                      "initial_momentum": "momentum",
                      "final_momentum": "momentum"}
    if param_mapping is not None:
        _param_mapping.update(param_mapping)

    name, args, kwargs = call
    for key in params:
        # If a parameter isn't named the same way in the call
        if key in _param_mapping:
            kwargs_key = _param_mapping[key]
        else:
            kwargs_key = key

        expected_value = params[key]
        actual_value = kwargs.get(kwargs_key, None)
        if expected_value != actual_value:
            raise AssertionError(
                "Mock not called with `%s=%s`. Called with `%s`" %
                (key, expected_value, actual_value)
            )


def check_mock_called_with_kwargs(mock: MagicMock, params: dict) -> None:
    """Check whether a mock was called with kwargs, but also some other params
    we don't care about."""
    assert len(mock.mock_calls) > 0, "Mock never called"
    for call in mock.mock_calls:
        check_call_contains_kwargs(call, params)


class TestTSNEParameterFlow(unittest.TestCase):
    """Test that the optimization parameters get properly propagated."""

    grad_descent_params = {
        "negative_gradient_method": [kl_divergence_bh, kl_divergence_fft],
        "learning_rate": [1, 10, 100],
        "dof": [0.5, 1, 1.5],
        "theta": [0.2, 0.5, 0.8],
        "n_interpolation_points": [3, 5],
        "min_num_intervals": [10, 20, 30],
        "ints_in_interval": [1, 2, 5],
        "max_grad_norm": [None, 0.5, 1],
        "max_step_norm": [None, 1, 5],
        "n_jobs": [1, 2, 4],
        "callbacks": [None, [lambda *args, **kwargs: ...]],
        "callbacks_every_iters": [25, 50],
    }

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

    @check_params({**grad_descent_params, **{
        "early_exaggeration_iter": [50, 100],
        "early_exaggeration": [4, 12],
        "initial_momentum": [0.2, 0.5, 0.8],
        "n_iter": [50, 100],
        "exaggeration": [None, 2],
        "final_momentum": [0.2, 0.5, 0.8],
    }})
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_constructor(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Early exaggeration training loop
        if param_name in ("early_exaggeration_iter", "early_exaggeration", "initial_momentum"):
            call_idx = 0
        # Main training loop
        elif param_name in ("n_iter", "exaggeration", "final_momentum"):
            call_idx = 1
        # If general parameter, should be applied to every call
        else:
            call_idx = 0

        TSNE(**{param_name: param_value}).fit(self.x)

        self.assertEqual(2, gradient_descent.call_count)
        check_call_contains_kwargs(
            gradient_descent.mock_calls[call_idx],
            {param_name: param_value},
        )

    @check_params({**grad_descent_params, **{
        "n_iter": [50, 100, 150],
        "exaggeration": [None, 2, 5],
        "momentum": [0.2, 0.5, 0.8],
    }})
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_embedding_optimize(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # `optimize` requires us to specify the `n_iter`
        params = {"n_iter": 50, param_name: param_value}

        tsne = TSNE()
        embedding = tsne.prepare_initial(self.x)
        embedding.optimize(**params, inplace=True)

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[0], params)

    @check_params({
        "early_exaggeration_iter": [50, 100],
        "early_exaggeration": [None, 2, 4],
        "n_iter": [50, 100],
        "exaggeration": [None, 1, 2],
        "initial_momentum": [0.2, 0.5, 0.8],
        "final_momentum": [0.2, 0.5, 0.8],
        "max_grad_norm": [None, 0.5, 1],
        "max_step_norm": [None, 1, 5],
    })
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_embedding_transform(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Perform initial embedding - this is tested above
        tsne = TSNE()
        embedding = tsne.fit(self.x)
        gradient_descent.reset_mock()

        embedding.transform(self.x_test, **{param_name: param_value})

        if "early" in param_name or "initial" in param_name:
            call_idx = 0
        else:
            call_idx = 1

        self.assertEqual(2, gradient_descent.call_count)
        check_call_contains_kwargs(
            gradient_descent.mock_calls[call_idx],
            {param_name: param_value},
        )

    @check_params({**grad_descent_params, **{
        "n_iter": [50, 100, 150],
        "exaggeration": [None, 2, 5],
        "momentum": [0.2, 0.5, 0.8],
        "max_grad_norm": [None, 0.5, 1],
        "max_step_norm": [None, 1, 5],
    }})
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_partial_embedding_optimize(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Perform initial embedding - this is tested above
        tsne = TSNE()
        embedding = tsne.fit(self.x)
        gradient_descent.reset_mock()

        # `optimize` requires us to specify the `n_iter`
        params = {"n_iter": 50, param_name: param_value}

        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding.optimize(**params, inplace=True)

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[0], params)

    @check_params({"metric": set(NNDescent.VALID_METRICS) - {"mahalanobis"}})
    @unittest.skipIf(not is_package_installed("pynndescent"), "`pynndescent`is not installed")
    @patch("pynndescent.NNDescent")
    def test_nndescent_distances(self, param_name, metric, nndescent: MagicMock):
        """Distance metrics should be properly passed down to NN descent"""
        assert param_name == "metric"
        tsne = TSNE(metric=metric, neighbors="pynndescent")

        # We don't care about what happens later, just that the NN method is
        # properly called
        nndescent.side_effect = InterruptedError()
        try:
            # Haversine distance only supports two dimensions
            tsne.prepare_initial(self.x[:, :2])
        except InterruptedError:
            pass

        self.assertEqual(nndescent.call_count, 1)
        check_call_contains_kwargs(nndescent.mock_calls[0], {"metric": metric})

    @unittest.skipIf(not is_package_installed("pynndescent"), "`pynndescent`is not installed")
    @patch("pynndescent.NNDescent")
    def test_nndescent_mahalanobis_distance(self, nndescent: MagicMock):
        """Distance metrics and additional params should be correctly passed down to NN descent"""
        metric = "mahalanobis"
        C = np.cov(self.x)
        tsne = TSNE(metric=metric, metric_params={"V": C}, neighbors="pynndescent")

        # We don't care about what happens later, just that the NN method is
        # properly called
        nndescent.side_effect = InterruptedError()
        try:
            tsne.prepare_initial(self.x)
        except InterruptedError:
            pass

        self.assertEqual(nndescent.call_count, 1)
        check_call_contains_kwargs(nndescent.mock_calls[0], {"metric": metric})

    def test_raises_error_on_unrecognized_metric(self):
        """Unknown distance metric should raise error"""
        tsne = TSNE(metric="imaginary", neighbors="exact")
        with self.assertRaises(ValueError):
            tsne.prepare_initial(self.x)

        tsne = TSNE(metric="imaginary", neighbors="approx")
        with self.assertRaises(ValueError):
            tsne.prepare_initial(self.x)

    def test_knn_kwargs(self):
        import sklearn

        with patch(
            "sklearn.neighbors.NearestNeighbors",
            wraps=sklearn.neighbors.NearestNeighbors,
        ) as mock:
            params = dict(algorithm="kd_tree", leaf_size=10)
            tsne = TSNE(neighbors="exact", knn_kwargs=params)
            tsne.fit(self.x)
            check_mock_called_with_kwargs(mock, params)


class TestTSNEInplaceOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        cls.x = datasets.load_iris()["data"]
        cls.x_test = cls.x[::3]
        cls.x_test += np.random.normal(0, 1, size=cls.x_test.shape)

    def test_embedding_inplace_optimization(self):
        embedding1 = self.tsne.prepare_initial(self.x)

        embedding2 = embedding1.optimize(n_iter=5, inplace=True)
        embedding3 = embedding2.optimize(n_iter=5, inplace=True)

        self.assertIs(embedding1.base, embedding2.base)
        self.assertIs(embedding2.base, embedding3.base)

    def test_embedding_not_inplace_optimization(self):
        embedding1 = self.tsne.prepare_initial(self.x)

        embedding2 = embedding1.optimize(n_iter=5, inplace=False)
        embedding3 = embedding2.optimize(n_iter=5, inplace=False)

        self.assertFalse(embedding1.base is embedding2.base)
        self.assertFalse(embedding2.base is embedding3.base)
        self.assertFalse(embedding1.base is embedding3.base)

    def test_partial_embedding_inplace_optimization(self):
        # Prepare reference embedding
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        partial_embedding1 = embedding.prepare_partial(self.x_test)
        partial_embedding2 = partial_embedding1.optimize(5, inplace=True)
        partial_embedding3 = partial_embedding2.optimize(5, inplace=True)

        self.assertIs(partial_embedding1.base, partial_embedding2.base)
        self.assertIs(partial_embedding2.base, partial_embedding3.base)

    def test_partial_embedding_not_inplace_optimization(self):
        # Prepare reference embedding
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        partial_embedding1 = embedding.prepare_partial(self.x_test)
        partial_embedding2 = partial_embedding1.optimize(5, inplace=False)
        partial_embedding3 = partial_embedding2.optimize(5, inplace=False)

        self.assertFalse(partial_embedding1.base is partial_embedding2.base)
        self.assertFalse(partial_embedding2.base is partial_embedding3.base)
        self.assertFalse(partial_embedding1.base is partial_embedding3.base)

    def test_inplace_embedding_optimization_doesnt_change_init(self):
        init = initialization.pca(self.x)
        init_copy = init.copy()

        aff = affinity.PerplexityBasedNN(self.x)
        embedding = TSNEEmbedding(init, aff)
        embedding.optimize(10, inplace=True)

        np.testing.assert_array_equal(init, init_copy)

    def test_inplace_partial_embedding_optimization_doesnt_change_init(self):
        embedding = TSNE(early_exaggeration_iter=10, n_iter=10).fit(self.x)

        init = initialization.random(self.x_test)
        init_copy = init.copy()

        P = embedding.affinities.to_new(self.x_test)

        partial_embedding = PartialTSNEEmbedding(init, embedding, P)
        partial_embedding.optimize(10, inplace=True)

        np.testing.assert_array_equal(init, init_copy)

    def test_inplace_partial_embedding_optimization_doesnt_change_embedding(self):
        embedding = TSNE(early_exaggeration_iter=10, n_iter=10).fit(self.x)
        embedding_copy = np.array(embedding)

        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding.optimize(10, inplace=True)

        np.testing.assert_array_equal(np.array(embedding), embedding_copy)


class TestTSNECallbackParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

    def test_can_pass_callbacks_to_tsne_object(self):
        callback = MagicMock()
        callback2 = MagicMock()
        # We don't want individual callbacks to be iterable
        del callback.__iter__
        del callback2.__iter__

        # Should be able to pass a single callback
        TSNE(callbacks=callback, callbacks_every_iters=1,
             early_exaggeration_iter=0, n_iter=1).fit(self.x)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        TSNE(callbacks=[callback], callbacks_every_iters=1,
             early_exaggeration_iter=0, n_iter=1).fit(self.x)
        self.assertEqual(callback.call_count, 1)

        # Should be able to change the callback on the object
        callback.reset_mock()
        tsne = TSNE(callbacks=callback, callbacks_every_iters=1,
                    early_exaggeration_iter=0, n_iter=1)
        tsne.callbacks = callback2
        tsne.fit(self.x)
        callback.assert_not_called()
        self.assertEqual(callback2.call_count, 1)

    def test_can_pass_callbacks_to_embedding_optimize(self):
        embedding = self.tsne.prepare_initial(self.x)

        # We don't the callback to be iterable
        callback = MagicMock()
        del callback.__iter__

        # Should be able to pass a single callback
        embedding.optimize(1, callbacks=callback, callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        embedding.optimize(1, callbacks=[callback], callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

    def test_can_pass_callbacks_to_partial_embedding_optimize(self):
        embedding = self.tsne.prepare_initial(self.x)

        # We don't the callback to be iterable
        callback = MagicMock()
        del callback.__iter__

        # Should be able to pass a single callback
        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding.optimize(1, callbacks=callback, callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        partial_embedding.optimize(1, callbacks=[callback], callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)


class TestAlternativeFitUsageWithAffinityAndInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.normal(100, 50, (25, 4))
        cls.init = np.random.normal(0, 1e-4, (25, 2))

    def test_fails_if_no_parameters_specified(self):
        tsne = TSNE()
        with self.assertRaises(ValueError):
            tsne.fit()

    def test_precomputed_affinity_is_passed_to_embedding_object(self):
        aff = affinity.PerplexityBasedNN(self.x, 5, method="exact")
        embedding = TSNE(
            early_exaggeration_iter=0, n_iter=0, initialization=self.init
        ).fit(affinities=aff)
        self.assertIs(embedding.affinities, aff)

    def test_fails_if_affinities_parameter_is_not_correct_class(self):
        aff = "definitely not an affinity object"
        with self.assertRaises(ValueError):
            TSNE(initialization=self.init).fit(affinities=aff)

    def test_precomputed_initialization_is_passed_to_embedding_object(self):
        embedding = TSNE(early_exaggeration_iter=0, n_iter=0) \
            .fit(self.x, initialization=self.init)
        np.testing.assert_array_equal(embedding, self.init)

    def test_string_initialization(self):
        # This should not crash
        TSNE(early_exaggeration_iter=0, n_iter=0).fit(self.x, initialization="pca")

    def test_parameter_init_takes_precendence_over_constructor_init(self):
        constructor_init = np.random.normal(1, 1e-4, self.init.shape)
        embedding = TSNE(
            early_exaggeration_iter=0, n_iter=0, initialization=constructor_init
        ).fit(self.x, initialization=self.init)
        np.testing.assert_array_equal(embedding, self.init)

    def test_pca_init_with_only_affinities_passed(self):
        aff = affinity.PerplexityBasedNN(self.x, 5, method="exact")
        desired_init = initialization.spectral(aff.P, random_state=42)
        embedding = TSNE(
            early_exaggeration_iter=0, n_iter=0, initialization="pca", random_state=42
        ).fit(affinities=aff)
        # The two spectral decompositions are run separately, and need only
        # agree to within the eigensolver's reproducibility
        np.testing.assert_allclose(embedding, desired_init)


class TSNEInitialization(unittest.TestCase):
    transform_initializations = ["random", "median", "weighted"]

    @classmethod
    def setUpClass(cls):
        # It would be nice if the initial data were not nicely behaved to test
        # for low variance
        cls.x = np.random.normal(100, 50, (25, 4))
        cls.iris = datasets.load_iris()["data"]

    def test_low_variance(self):
        """Low variance of the initial embedding is very important for the
        convergence of tSNE."""
        # Cycle through various initializations
        initializations = ["random", "pca"]
        allowed = 1e-3

        for init in initializations:
            tsne = TSNE(initialization=init, perplexity=2)
            embedding = tsne.prepare_initial(self.x)
            np.testing.assert_array_less(np.var(embedding, axis=0), allowed,
                                         "using the `%s` initialization" % init)

    def test_mismatching_embedding_dimensions_simple_api(self):
        # Fit
        tsne = TSNE(n_components=2, initialization=self.x[:10, :2])
        with self.assertRaises(ValueError, msg="fit::incorrect number of points"):
            tsne.fit(self.x[:25])

        with self.assertRaises(ValueError, msg="fit::incorrect number of dimensions"):
            TSNE(n_components=2, initialization=self.x[:10, :4])

        # Transform
        tsne = TSNE(n_components=2, initialization="random")
        embedding = tsne.fit(self.x)
        with self.assertRaises(ValueError, msg="transform::incorrect number of points"):
            embedding.transform(X=self.x[:5], initialization=self.x[:10, :2])

        with self.assertRaises(ValueError, msg="transform::incorrect number of dimensions"):
            embedding.transform(X=self.x, initialization=self.x[:, :4])

    def test_same_unoptimized_initializations_for_transform(self):
        """Initializations should be deterministic."""
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        embedding = openTSNE.TSNE(
            early_exaggeration_iter=50,
            n_iter=50,
            neighbors="exact",
            negative_gradient_method="bh",
            random_state=42,
        ).fit(x_train)

        for init in self.transform_initializations:
            new_embedding_1 = embedding.prepare_partial(x_test, initialization=init)
            new_embedding_2 = embedding.prepare_partial(x_test, initialization=init)

            np.testing.assert_equal(new_embedding_1, new_embedding_2, init)

    def test_same_bh_optimized_median_initializations_for_transform(self):
        """Transform with Barnes-Hut optimization should be deterministic."""
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        embedding = openTSNE.TSNE(
            early_exaggeration_iter=10,
            n_iter=10,
            neighbors="exact",
            negative_gradient_method="bh",
            random_state=42,
        ).fit(x_train)

        for init in self.transform_initializations:
            new_embedding_1 = embedding.transform(
                x_test, initialization=init, n_iter=10
            )
            new_embedding_2 = embedding.transform(
                x_test, initialization=init, n_iter=10
            )

            np.testing.assert_equal(new_embedding_1, new_embedding_2, init)

    def test_same_fft_optimized_median_initializations_for_transform(self):
        """Transform with interpolation based optimization should be deterministic."""
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        embedding = openTSNE.TSNE(
            early_exaggeration_iter=10,
            n_iter=10,
            neighbors="exact",
            negative_gradient_method="fft",
            random_state=42,
        ).fit(x_train)

        for init in self.transform_initializations:
            new_embedding_1 = embedding.transform(
                x_test, initialization=init, n_iter=10, learning_rate=10
            )
            new_embedding_2 = embedding.transform(
                x_test, initialization=init, n_iter=10, learning_rate=10
            )

            np.testing.assert_equal(new_embedding_1, new_embedding_2, init)


class TestRandomState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # It would be nice if the initial data were not nicely behaved to test
        # for low variance
        cls.x = np.random.normal(10000, 50, (25, 4))
        cls.x_test = np.random.normal(100, 50, (25, 4))

    def test_same_results_on_fixed_random_state_random_init(self):
        """Results should be exactly the same if we provide a random state."""
        tsne1 = TSNE(random_state=1, initialization="random")
        embedding1 = tsne1.fit(self.x)

        tsne2 = TSNE(random_state=1, initialization="random")
        embedding2 = tsne2.fit(self.x)

        np.testing.assert_array_equal(
            embedding1,
            embedding2,
            "Same random state produced different initial embeddings",
        )

    def test_same_results_on_fixed_random_state_pca_init(self):
        """Results should be exactly the same if we provide a random state."""
        tsne1 = TSNE(random_state=1, initialization="pca")
        embedding1 = tsne1.fit(self.x)

        tsne2 = TSNE(random_state=1, initialization="pca")
        embedding2 = tsne2.fit(self.x)

        np.testing.assert_array_equal(
            embedding1,
            embedding2,
            "Same random state produced different initial embeddings",
        )

    def test_same_partial_embedding_on_fixed_random_state(self):
        tsne = TSNE(random_state=1, initialization="random")
        embedding = tsne.fit(self.x)

        partial1 = embedding.prepare_partial(self.x_test, initialization="random")
        partial2 = embedding.prepare_partial(self.x_test, initialization="random")

        np.testing.assert_array_equal(
            partial1,
            partial2,
            "Same random state produced different partial embeddings",
        )

    @patch("openTSNE.initialization.random", wraps=openTSNE.initialization.random)
    @patch("openTSNE.nearest_neighbors.Sklearn", wraps=openTSNE.nearest_neighbors.Sklearn)
    def test_random_state_parameter_is_propagated_random_init_exact(self, init, neighbors):
        random_state = 1

        tsne = openTSNE.TSNE(
            neighbors="exact",
            initialization="random",
            random_state=random_state,
        )
        tsne.prepare_initial(self.x)

        # Verify that `random_state` was passed
        init.assert_called_once()
        check_mock_called_with_kwargs(init, {"random_state": random_state})
        neighbors.assert_called_once()
        check_mock_called_with_kwargs(neighbors, {"random_state": random_state})

    @patch("openTSNE.initialization.pca", wraps=openTSNE.initialization.pca)
    @patch("openTSNE.nearest_neighbors.Annoy", wraps=openTSNE.nearest_neighbors.Annoy)
    def test_random_state_parameter_is_propagated_pca_init_approx(self, init, neighbors):
        random_state = 1

        tsne = openTSNE.TSNE(
            neighbors="approx",
            initialization="pca",
            random_state=random_state,
        )
        tsne.prepare_initial(self.x)

        # Verify that `random_state` was passed
        init.assert_called_once()
        check_mock_called_with_kwargs(init, {"random_state": random_state})
        neighbors.assert_called_once()
        check_mock_called_with_kwargs(neighbors, {"random_state": random_state})


class TestDefaultParameterSettings(unittest.TestCase):
    def test_default_params_simple_vs_complex_flow(self):
        # Relevant affinity parameters are passed to the affinity object
        mismatching = get_mismatching_default_values(
            openTSNE.TSNE,
            PerplexityBasedNN,
            {"neighbors": "method"},
        )
        self.assertEqual(mismatching, [])

        assert len(
            get_shared_parameters(openTSNE.TSNE, openTSNE.tsne.gradient_descent.__call__)
        ) > 0, \
            "`TSNE` and `gradient_descent` have no shared parameters. Have you " \
            "changed the signature or usage?"

        # The relevant gradient descent parameters are passed down directly to
        # `gradient_descent`
        mismatching = get_mismatching_default_values(
            openTSNE.TSNE,
            openTSNE.tsne.gradient_descent.__call__,
        )
        # Some default parameters should be different between TSNE and gradient_descent
        allowed_mismatches = ("n_iter", "learning_rate")
        mismatching = list(filter(lambda x: x[0] not in allowed_mismatches, mismatching))
        self.assertEqual(mismatching, [])


def get_shared_parameters(f1, f2):
    """Get the names of shared parameters from two function signatures."""
    params1 = inspect.signature(f1).parameters
    params2 = inspect.signature(f2).parameters

    return set(params1.keys()) & set(params2.keys())


def get_mismatching_default_values(f1, f2, mapping=None):
    """Check that two functions have the same default values for shared parameters."""
    # Additional mappings from f1 parameters to f2 parameters may be provided
    if mapping is None:
        mapping = {}

    params1 = inspect.signature(f1).parameters
    params2 = inspect.signature(f2).parameters

    mismatch = []
    for f1_param_name in params1:
        # If the param is named differently in f2, rename
        f2_param_name = mapping[f1_param_name] if f1_param_name in mapping else f1_param_name

        # If the parameter does not appear in the signature of f2, there"s
        # nothing to do
        if f2_param_name not in params2:
            continue

        val1 = params1[f1_param_name].default
        val2 = params2[f2_param_name].default

        if val1 != val2:
            mismatch.append((f1_param_name, val1, f2_param_name, val2))

    return mismatch


class TestGradientResult(unittest.TestCase):
    def test_required_fields(self):
        g = np.zeros((4, 2))
        r = GradientResult(error=1.5, gradient=g)
        self.assertEqual(r.error, 1.5)
        self.assertIs(r.gradient, g)
        self.assertEqual(r.dof_grad, 0.0)

    def test_dof_grad_can_be_set(self):
        r = GradientResult(error=0.0, gradient=np.zeros((2, 2)), dof_grad=-0.7)
        self.assertEqual(r.dof_grad, -0.7)


class TestGradientDescentOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        random_state = np.random.RandomState(42)
        cls.x = random_state.randn(100, 4)
        cls.x_test = random_state.randn(25, 4)

    def test_optimizer_being_passed_to_subsequent_embeddings(self):
        embedding = self.tsne.prepare_initial(self.x)

        self.assertIsNone(
            embedding.optimizer.optimizers["embedding"].gains, "Optimizer should be initialized with no gains"
        )

        # Check the switch from no gains to some gains
        embedding1 = embedding.optimize(10)
        self.assertIsNone(
            embedding.optimizer.optimizers["embedding"].gains,
            "Gains changed on initial optimizer even though we did not do "
            "inplace optimization.",
        )
        self.assertIsNotNone(
            embedding1.optimizer.optimizers["embedding"].gains, "Gains were not properly set in new embedding."
        )
        self.assertIsNot(
            embedding.optimizer,
            embedding1.optimizer,
            "The embedding and new embedding optimizer are the same instance "
            "even we did not do inplace optimization.",
        )

        # Check switch from existing gains to new gains
        embedding2 = embedding1.optimize(10)
        self.assertIsNot(
            embedding1.optimizer,
            embedding2.optimizer,
            "The embedding and new embedding optimizer are the same instance "
            "even we did not do inplace optimization.",
        )
        self.assertFalse(
            np.allclose(embedding1.optimizer.optimizers["embedding"].gains, embedding2.optimizer.optimizers["embedding"].gains),
            "The gains in the new embedding did not change at all from the old "
            "embedding.",
        )

    def test_optimizer_being_passed_to_partial_embeddings(self):
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        # Partial embeddings get their own optimizer instance
        partial = embedding.prepare_partial(self.x_test)
        self.assertIsNot(
            embedding.optimizer,
            partial.optimizer,
            "Embedding and partial embedding optimizers are the same instance.",
        )
        self.assertIsNone(
            partial.optimizer.optimizers["embedding"].gains,
            "Partial embedding was not initialized with no gains",
        )

        # Check the switch from no gains to some gains
        partial1 = partial.optimize(10)
        self.assertIsNone(
            partial.optimizer.optimizers["embedding"].gains,
            "Gains on initial optimizer changed even though we did not do "
            "inplace optimization.",
        )
        self.assertIsNotNone(
            partial1.optimizer.optimizers["embedding"].gains,
            "Gains were not properly set in new partial embedding.",
        )

        # Check switch from existing gains to new gains
        partial2 = partial1.optimize(10)
        self.assertIsNot(
            partial1.optimizer,
            partial2.optimizer,
            "The embedding and new embedding optimizer are the same instance "
            "even we did not do inplace optimization.",
        )
        self.assertFalse(
            np.allclose(partial1.optimizer.optimizers["embedding"].gains, partial2.optimizer.optimizers["embedding"].gains),
            "The gains in the new embedding did not change at all from the old "
            "embedding.",
        )

    def test_embedding_optimizer_inplace(self):
        embedding0 = self.tsne.prepare_initial(self.x)

        # Assign only so the references are clear
        embedding1 = embedding0.optimize(10, inplace=True)
        embedding2 = embedding1.optimize(10, inplace=True)

        self.assertIs(embedding0.optimizer, embedding1.optimizer)
        self.assertIs(embedding1.optimizer, embedding2.optimizer)

    def test_partial_embedding_optimizer_inplace(self):
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)
        partial0 = embedding.prepare_partial(self.x_test)

        # Assign only so the references are clear
        partial1 = partial0.optimize(10, inplace=True)
        partial2 = partial1.optimize(10, inplace=True)

        self.assertIs(partial0.optimizer, partial1.optimizer)
        self.assertIs(partial1.optimizer, partial2.optimizer)

    def test_pickling(self):
        obj = openTSNE.tsne.gradient_descent()
        obj.optimizers["embedding"].gains = np.ones(5)
        loaded_obj = pickle.loads(pickle.dumps(obj))
        np.testing.assert_array_equal(
            loaded_obj.optimizers["embedding"].gains, np.ones(5)
        )

    def test_unpickling_legacy_optimizer_state(self):
        # Pickles produced before the optimizer refactor stored `gains` and
        # `update` directly on `gradient_descent` (and on the learnable-dof
        # branch, also `dof_gain`/`dof_update`). Round-trip both shapes
        # through pickle by setting the legacy attrs on the instance dict
        # before dumping.
        from openTSNE.optimizer import DeltaBarDeltaOptimizer

        legacy_master = openTSNE.tsne.gradient_descent()
        legacy_master.__dict__ = {"gains": np.ones(5), "update": np.zeros(5)}
        loaded = pickle.loads(pickle.dumps(legacy_master))
        self.assertIsInstance(
            loaded.optimizers["embedding"], DeltaBarDeltaOptimizer
        )
        np.testing.assert_array_equal(
            loaded.optimizers["embedding"].gains, np.ones(5)
        )
        np.testing.assert_array_equal(
            loaded.optimizers["embedding"].update, np.zeros(5)
        )
        self.assertNotIn("dof", loaded.optimizers)

    def test_gains_is_always_numpy_array(self):
        embedding = self.tsne.prepare_initial(self.x)
        self.assertIsInstance(embedding.optimizer.optimizers["embedding"].gains, (type(None), np.ndarray))
        self.assertNotIsInstance(embedding.optimizer.optimizers["embedding"].gains, openTSNE.TSNEEmbedding)

        embedding = embedding.optimize(10)
        self.assertIsInstance(embedding.optimizer.optimizers["embedding"].gains, (type(None), np.ndarray))
        self.assertNotIsInstance(embedding.optimizer.optimizers["embedding"].gains, openTSNE.TSNEEmbedding)

        embedding.optimize(10, inplace=True)
        self.assertIsInstance(embedding.optimizer.optimizers["embedding"].gains, (type(None), np.ndarray))
        self.assertNotIsInstance(embedding.optimizer.optimizers["embedding"].gains, openTSNE.TSNEEmbedding)

    def test_pickling_via_embedding(self):
        embedding = self.tsne.prepare_initial(self.x)
        # Before optimization
        loaded_embedding = pickle.loads(pickle.dumps(embedding))
        np.testing.assert_equal(
            embedding.optimizer.optimizers["embedding"].gains,
            loaded_embedding.optimizer.optimizers["embedding"].gains,
            "Failed loading without any optimization",
        )

        embedding = embedding.optimize(10)

        # After optimization
        loaded_embedding = pickle.loads(pickle.dumps(embedding))
        np.testing.assert_equal(
            embedding.optimizer.optimizers["embedding"].gains,
            loaded_embedding.optimizer.optimizers["embedding"].gains,
            "Failed loading after optimization (differing gains)",
        )

    def test_gains_are_kept_for_points_inside_interpolation_grid(self):
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        partial = embedding.prepare_partial(self.x_test)
        partial = partial.optimize(10, negative_gradient_method="fft", learning_rate=0.1)

        self.assertTrue(
            np.all(partial.optimizer.optimizers["embedding"].gains > 0),
            "Gains of points inside the interpolation grid were zeroed out.",
        )

    def test_gains_are_zeroed_only_for_points_hitting_grid_boundary(self):
        for n_components in (1, 2):
            tsne = TSNE(n_components=n_components, negative_gradient_method="fft")
            embedding = tsne.prepare_initial(self.x)
            embedding.optimize(10, inplace=True)

            # Start the first point far outside of the interpolation grid, so
            # it is clipped to the boundary in every iteration
            initialization = np.array(embedding.prepare_partial(self.x_test))
            initialization[0] = 1e3
            partial = embedding.prepare_partial(
                self.x_test, initialization=initialization
            )
            partial = partial.optimize(5, learning_rate=0.1)

            gains = partial.optimizer.optimizers["embedding"].gains
            np.testing.assert_array_equal(
                gains[0], 0, "%dd: point at the boundary kept its gains" % n_components
            )
            self.assertTrue(
                np.all(gains[1:] > 0),
                "%dd: points inside the grid had their gains zeroed" % n_components,
            )


class TestDeltaBarDeltaOptimizer(unittest.TestCase):
    def test_defaults_match_historical_constants(self):
        from openTSNE.optimizer import DeltaBarDeltaOptimizer

        opt = DeltaBarDeltaOptimizer()
        self.assertEqual(opt.gain_increase, 0.2)
        self.assertEqual(opt.gain_decay, 0.8)
        self.assertEqual(opt.max_gain, np.inf)

    def test_gain_runs_away_on_monotone_gradient_when_uncapped(self):
        # A constant-sign gradient makes every step a consistent descent, so the
        # uncapped gain grows ~ 1 + gain_increase * iters.
        from openTSNE.optimizer import DeltaBarDeltaOptimizer

        opt = DeltaBarDeltaOptimizer()
        for _ in range(100):
            opt.step(np.array([1.0]), learning_rate=0.01, momentum=0.0)
        self.assertGreater(opt.gains[0], 15.0)

    def test_max_gain_caps_runaway(self):
        from openTSNE.optimizer import DeltaBarDeltaOptimizer

        opt = DeltaBarDeltaOptimizer(max_gain=1.0)
        for _ in range(100):
            opt.step(np.array([1.0]), learning_rate=0.01, momentum=0.0)
        self.assertLessEqual(opt.gains[0], 1.0)

    def test_dof_optimizer_is_gain_capped(self):
        # The dof optimizer must be created with a capped gain so the scalar dof
        # cannot run away.
        tsne = openTSNE.TSNE(
            neighbors="exact", negative_gradient_method="bh",
            dof="auto", random_state=42, n_iter=10, early_exaggeration_iter=10,
        )
        emb = tsne.fit(datasets.load_iris()["data"])
        self.assertEqual(emb.optimizer.optimizers["dof"].max_gain, 1.0)

    def test_setstate_supplies_missing_gain_params(self):
        # Pickles from before the gain params were stored must still load.
        from openTSNE.optimizer import DeltaBarDeltaOptimizer

        opt = DeltaBarDeltaOptimizer()
        opt.__setstate__({"gains": np.ones(3), "update": np.zeros(3)})
        self.assertEqual(opt.gain_increase, 0.2)
        self.assertEqual(opt.gain_decay, 0.8)
        self.assertEqual(opt.max_gain, np.inf)


class TestAffinityIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # It would be nice if the initial data were not nicely behaved to test
        # for low variance
        cls.x = np.random.normal(100, 50, (25, 4))
        cls.x_test = np.random.normal(100, 50, (25, 4))

    def test_transform_with_standard_affinity(self):
        init = openTSNE.initialization.random(self.x)
        aff = openTSNE.affinity.PerplexityBasedNN(self.x, 5, method="exact")
        embedding = openTSNE.TSNEEmbedding(init, aff, negative_gradient_method="bh")
        embedding.optimize(100, inplace=True)

        # This should not raise an error
        embedding.transform(self.x_test)

    def test_transform_with_multiscale_affinity(self):
        init = openTSNE.initialization.random(self.x)
        aff = openTSNE.affinity.Multiscale(self.x, [2, 5], method="exact")
        embedding = openTSNE.TSNEEmbedding(init, aff, negative_gradient_method="bh")
        embedding.optimize(100, inplace=True)

        # This should not raise an error
        embedding.transform(self.x_test)

    def test_transform_with_nonstandard_affinity(self):
        """Should raise an informative error when a non-standard affinity is used
        with `transform`."""
        init = openTSNE.initialization.random(self.x)
        aff = openTSNE.affinity.Uniform(self.x, 5, method="exact")
        embedding = openTSNE.TSNEEmbedding(init, aff, negative_gradient_method="bh")
        embedding.optimize(100, inplace=True)

        with self.assertRaises(TypeError):
            embedding.transform(self.x_test)


class TestTSNEEmebedding(unittest.TestCase):
    def test_pickling(self):
        tsne = TSNE(random_state=4)
        embedding = tsne.fit(np.random.randn(100, 4))
        loaded_obj = pickle.loads(pickle.dumps(embedding))
        self.assertIsInstance(loaded_obj, openTSNE.TSNEEmbedding)
        self.assertIsInstance(loaded_obj.affinities, openTSNE.affinity.Affinities)
        self.assertEqual(4, loaded_obj.random_state)

    def test_pickling_with_transform(self):
        tsne = TSNE(random_state=4)
        embedding: openTSNE.TSNEEmbedding = tsne.fit(np.random.randn(100, 4))
        loaded_obj: openTSNE.TSNEEmbedding = pickle.loads(pickle.dumps(embedding))
        loaded_obj.transform(np.random.randn(100, 4))

    @unittest.skipIf(not is_package_installed("hnswlib"), "`hnswlib` is not installed")
    def test_pickling_with_transform_hnsw(self):
        x, x_new = np.random.randn(100, 4), np.random.randn(20, 4)
        tsne = TSNE(neighbors="hnsw", random_state=4)
        embedding: openTSNE.TSNEEmbedding = tsne.fit(x)
        loaded_obj: openTSNE.TSNEEmbedding = pickle.loads(pickle.dumps(embedding))
        self.assertIsInstance(
            loaded_obj.affinities.knn_index, openTSNE.nearest_neighbors.HNSW
        )

        np.testing.assert_array_almost_equal(
            loaded_obj.transform(x_new), embedding.transform(x_new)
        )

    def test_pickling_preserves_optimization_iters(self):
        tsne = TSNE(early_exaggeration_iter=3, n_iter=7, random_state=4)
        embedding = tsne.fit(np.random.randn(100, 4))
        self.assertEqual(embedding.optimization_iters_, 10)

        loaded = pickle.loads(pickle.dumps(embedding))
        self.assertEqual(loaded.optimization_iters_, 10)


class TestPrecomputedDistanceMatrices(unittest.TestCase):
    def test_precomputed_dist_matrix_via_affinities_uses_spectral_init(self):
        x = np.random.normal(0, 1, (200, 5))
        d = squareform(pdist(x))

        aff = affinity.PerplexityBasedNN(d, metric="precomputed")
        desired_init = initialization.spectral(aff.P, random_state=42)
        embedding = TSNE(
            early_exaggeration_iter=0, 
            n_iter=0, 
            random_state=42,
        ).fit(affinities=aff)
        # See note in `test_pca_init_with_only_affinities_passed`: the spectral
        # init is recomputed inside `fit`, so compare with a tolerance.
        np.testing.assert_allclose(embedding, desired_init, rtol=1e-5, atol=1e-10)

    def test_precomputed_dist_matrix_via_tsne_interface_uses_spectral_init(self):
        x = np.random.normal(0, 1, (200, 5))
        d = squareform(pdist(x))

        aff = affinity.PerplexityBasedNN(d, metric="precomputed")
        desired_init = initialization.spectral(aff.P, random_state=42)
        embedding = TSNE(
            metric="precomputed", 
            early_exaggeration_iter=0, 
            n_iter=0,
            random_state=42,
         ).fit(d)
        # See note in `test_pca_init_with_only_affinities_passed`: the spectral
        # init is recomputed inside `fit`, so compare with a tolerance.
        np.testing.assert_allclose(embedding, desired_init, rtol=1e-5, atol=1e-10)

    def test_precomputed_dist_matrix_doesnt_override_valid_inits(self):
        iris = datasets.load_iris()
        x, y = iris.data, iris.target
        d = squareform(pdist(x))

        embedding = TSNE(
            initialization="random",
            metric="precomputed",
            early_exaggeration_iter=0,
            n_iter=0,
        ).fit(d)

        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(embedding, y)
        predictions = knn.predict(embedding)
        self.assertLess(accuracy_score(predictions, y), 0.6)


class TestMisc(unittest.TestCase):
    def test_very_large_affinity_matrices(self):
        x = np.random.normal(0, 1, (50, 10))
        aff = PerplexityBasedNN(x, perplexity=30)

        # Super large affinity matrices have so many indices, it needs to be
        # stored as long
        aff.P.indptr = aff.P.indptr.astype(np.int64)
        aff.P.indices = aff.P.indices.astype(np.int64)

        TSNE().fit(x, affinities=aff)

        # The old version should still work
        aff.P.indptr = aff.P.indptr.astype(np.int32)
        aff.P.indices = aff.P.indices.astype(np.int32)

        TSNE().fit(x, affinities=aff)

    def test_interpolation_grid_not_called_using_bh(self):
        x1 = np.random.normal(0, 1, (50, 10))
        x2 = np.random.normal(0, 1, (20, 10))

        with patch("openTSNE.TSNEEmbedding.prepare_interpolation_grid") as prep_grid:
            tsne = openTSNE.TSNE(negative_gradient_method="bh")
            embedding = tsne.fit(x1)
            # Calling transform shouldn't call `prepare_interpolation_grid`
            embedding.transform(x2)

            prep_grid.assert_not_called()
