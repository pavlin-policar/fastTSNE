"""Tests for the learnable degrees-of-freedom feature."""
import logging
import unittest
import warnings
from functools import partial

import numpy as np
from sklearn import datasets

import openTSNE
from openTSNE import affinity
from openTSNE.tsne import (
    GradientResult,
    IterationState,
    kl_divergence_bh,
    kl_divergence_fft,
)

np.random.seed(42)
affinity.log.setLevel(logging.ERROR)

TSNE_BH = partial(
    openTSNE.TSNE, neighbors="exact", negative_gradient_method="bh", random_state=42
)
TSNE_FFT = partial(
    openTSNE.TSNE, neighbors="exact", negative_gradient_method="fft", random_state=42
)


class _ObjectiveFnMixin:
    """Build a small (embedding, P) pair the objective functions can chew on."""

    @classmethod
    def setUpClass(cls):
        x = datasets.load_iris()["data"]
        aff = affinity.PerplexityBasedNN(x, perplexity=30, random_state=42)
        cls.P = aff.P
        cls.embedding = np.random.RandomState(0).randn(x.shape[0], 2) * 1e-4
        cls.bh_params = {"theta": 0.5}
        cls.fft_params = {
            "n_interpolation_points": 3,
            "min_num_intervals": 10,
            "ints_in_interval": 1,
        }


class TestKLDivergenceBHReturnType(_ObjectiveFnMixin, unittest.TestCase):
    def test_returns_gradient_result(self):
        result = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            should_eval_error=True,
        )
        self.assertIsInstance(result, GradientResult)
        self.assertEqual(result.gradient.shape, self.embedding.shape)
        self.assertTrue(np.isfinite(result.error))

    def test_compute_dof_grad_false_yields_zero(self):
        result = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=False,
        )
        self.assertEqual(result.dof_grad, 0.0)

    def test_compute_dof_grad_true_yields_nonzero(self):
        # For a non-trivial random embedding far from optimum, the dof gradient
        # should be measurably away from 0.
        result = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=True,
        )
        self.assertNotEqual(result.dof_grad, 0.0)
        self.assertTrue(np.isfinite(result.dof_grad))

    def test_gradient_unaffected_by_compute_dof_grad(self):
        # Toggling the flag must not change the embedding gradient itself.
        r1 = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=False,
        )
        r2 = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=True,
        )
        np.testing.assert_allclose(r1.gradient, r2.gradient)

    def test_dof_grad_matches_finite_difference(self):
        """Numerical correctness check: dof_grad must match a central-difference
        estimate of dKL/d(dof) computed at the same embedding.

        The analytical and finite-difference paths both use the same Barnes-Hut
        approximation, so the BH error cancels out and only the finite-difference
        truncation error matters. P is normalized (sum_P = 1, no exaggeration),
        which is the regime where the implementation's asymmetric normalization
        is consistent with the true gradient.
        """
        rng = np.random.RandomState(7)
        # Use a non-tiny embedding so dof_grad has meaningful magnitude.
        embedding = rng.randn(self.embedding.shape[0], 2) * 0.5
        dof = 1.0
        h = 1e-4

        analytical = kl_divergence_bh(
            embedding.copy(),
            self.P,
            dof=dof,
            bh_params=self.bh_params,
            should_eval_error=True,
            compute_dof_grad=True,
        ).dof_grad

        kl_plus = kl_divergence_bh(
            embedding.copy(),
            self.P,
            dof=dof + h,
            bh_params=self.bh_params,
            should_eval_error=True,
        ).error
        kl_minus = kl_divergence_bh(
            embedding.copy(),
            self.P,
            dof=dof - h,
            bh_params=self.bh_params,
            should_eval_error=True,
        ).error
        finite_diff = (kl_plus - kl_minus) / (2 * h)

        # Sanity: gradient should be non-trivial; otherwise the test is vacuous.
        self.assertGreater(abs(analytical), 1e-3)

        rel_err = abs(analytical - finite_diff) / abs(analytical)
        self.assertLess(
            rel_err,
            5e-3,
            f"dof_grad={analytical:.6g}, finite_diff={finite_diff:.6g}, "
            f"rel_err={rel_err:.3g}",
        )


class TestKLDivergenceFFTReturnType(_ObjectiveFnMixin, unittest.TestCase):
    def test_returns_gradient_result(self):
        result = kl_divergence_fft(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            fft_params=self.fft_params,
            should_eval_error=True,
        )
        self.assertIsInstance(result, GradientResult)
        self.assertEqual(result.gradient.shape, self.embedding.shape)

    def test_dof_grad_always_zero(self):
        # FFT objective does not compute dof_grad; it must be 0 regardless of
        # whether the caller asked for it.
        for flag in (False, True):
            with self.subTest(compute_dof_grad=flag):
                result = kl_divergence_fft(
                    self.embedding.copy(),
                    self.P,
                    dof=1.0,
                    fft_params=self.fft_params,
                    compute_dof_grad=flag,
                )
                self.assertEqual(result.dof_grad, 0.0)


class TestOptimizerReturnShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_fit_returns_embedding_with_kl(self):
        emb = TSNE_BH(early_exaggeration_iter=5, n_iter=5).fit(self.x)
        self.assertTrue(hasattr(emb, "kl_divergence"))
        self.assertTrue(np.isfinite(emb.kl_divergence))


class TestDofAutoLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_bh_auto_actually_moves_dof(self):
        # With dof="auto" on the BH path, dof should change over iterations.
        history = []
        TSNE_BH(
            dof="auto",
            initial_dof=1.0,
            early_exaggeration_iter=0,
            n_iter=30,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        dofs = [s.dof for s in history]
        self.assertGreater(len(dofs), 0)
        self.assertNotEqual(dofs[0], dofs[-1])

    def test_default_dof_no_initial_dof(self):
        # Path 1: dof=1 (default), initial_dof=None. Dof should stay at 1
        # throughout, no warnings about initial_dof.
        history = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                early_exaggeration_iter=0,
                n_iter=10,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(all(s.dof == 1.0 for s in history))
        self.assertFalse(
            any("initial_dof" in str(w.message) for w in caught),
            f"Did not expect initial_dof warning, got: {[str(w.message) for w in caught]}",
        )

    def test_fixed_dof_with_initial_dof_warns(self):
        # Path 2: dof=1 (fixed), initial_dof=5. initial_dof must be ignored
        # and a warning must be emitted.
        history = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                initial_dof=5.0,
                early_exaggeration_iter=0,
                n_iter=10,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(
            any("initial_dof" in str(w.message) for w in caught),
            f"Expected initial_dof warning, got: {[str(w.message) for w in caught]}",
        )
        self.assertTrue(all(s.dof == 1.0 for s in history))

    def test_nondefault_fixed_dof_with_initial_dof_warns(self):
        # Path 2b: dof=2.5 (non-default fixed), initial_dof=5. Same warning;
        # dof stays at 2.5.
        history = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                dof=2.5,
                initial_dof=5.0,
                early_exaggeration_iter=0,
                n_iter=10,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(
            any("initial_dof" in str(w.message) for w in caught),
            f"Expected initial_dof warning, got: {[str(w.message) for w in caught]}",
        )
        self.assertTrue(all(s.dof == 2.5 for s in history))

    def test_auto_dof_no_initial_dof_starts_from_one(self):
        # Path 3: dof="auto", initial_dof=None. Starting dof must default to 1.
        history = []
        TSNE_BH(
            dof="auto",
            early_exaggeration_iter=0,
            n_iter=2,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertGreater(len(history), 0)
        # First iteration's `dof` is the starting value before any update.
        self.assertEqual(history[0].dof, 1.0)

    def test_auto_dof_with_initial_dof_starts_from_initial(self):
        # Path 4: dof="auto", initial_dof=5. Starting dof must be 5.
        history = []
        TSNE_BH(
            dof="auto",
            initial_dof=5.0,
            early_exaggeration_iter=0,
            n_iter=2,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0].dof, 5.0)

    def test_fft_auto_warns_and_keeps_dof_fixed(self):
        # FFT path cannot learn dof; we must warn and dof must remain fixed.
        history = []
        with self.assertLogs("openTSNE.tsne", level="WARNING") as cm:
            TSNE_FFT(
                dof="auto",
                initial_dof=1.0,
                early_exaggeration_iter=0,
                n_iter=20,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(
            any("dof='auto'" in msg or "Barnes-Hut" in msg for msg in cm.output),
            f"Expected dof-auto warning, got: {cm.output}",
        )
        self.assertTrue(all(s.dof == 1.0 for s in history))


class TestIterationStateCallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_callback_receives_iteration_state(self):
        seen = []
        TSNE_BH(
            early_exaggeration_iter=0,
            n_iter=5,
            callbacks=seen.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertEqual(len(seen), 5)
        for state in seen:
            self.assertIsInstance(state, IterationState)
            self.assertEqual(state.embedding.shape[1], 2)
            self.assertEqual(state.gradient.shape, state.embedding.shape)
            self.assertTrue(np.isfinite(state.error))

    def test_callback_state_is_a_copy(self):
        # Callback must receive snapshots so retained state doesn't mutate.
        seen = []
        TSNE_BH(
            early_exaggeration_iter=0,
            n_iter=3,
            callbacks=seen.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        # Embeddings on consecutive iters must differ (optimizer moved).
        self.assertFalse(np.array_equal(seen[0].embedding, seen[-1].embedding))
        # And the snapshot arrays must not alias each other.
        self.assertIsNot(seen[0].embedding, seen[-1].embedding)

    def test_old_three_arg_callback_still_works_with_warning(self):
        calls = []

        def old_callback(iteration, error, embedding):
            calls.append((iteration, error, embedding.shape))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                early_exaggeration_iter=0,
                n_iter=5,
                callbacks=old_callback,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertEqual(len(calls), 5)
        self.assertTrue(
            any(issubclass(w.category, FutureWarning) for w in caught),
            f"Expected FutureWarning, got: {[w.category for w in caught]}",
        )

    def test_optimization_about_to_start_fires_for_new_style_callback(self):
        # Class-based callback with the new IterationState signature.
        class MyCallback:
            def __init__(self):
                self.start_calls = 0
                self.iter_calls = 0

            def optimization_about_to_start(self):
                self.start_calls += 1

            def __call__(self, state):
                self.iter_calls += 1

        cb = MyCallback()
        TSNE_BH(
            early_exaggeration_iter=5,
            n_iter=5,
            callbacks=cb,
            callbacks_every_iters=1,
        ).fit(self.x)
        # Standard fit calls .optimize() twice (early exaggeration + main).
        self.assertEqual(cb.start_calls, 2)
        self.assertGreater(cb.iter_calls, 0)

    def test_optimization_about_to_start_fires_for_old_style_callback(self):
        # Class-based callback wrapped by the deprecation adapter must still
        # have its `optimization_about_to_start` hook invoked.
        class MyOldCallback:
            def __init__(self):
                self.start_calls = 0
                self.iter_calls = 0

            def optimization_about_to_start(self):
                self.start_calls += 1

            def __call__(self, iteration, error, embedding):
                self.iter_calls += 1

        cb = MyOldCallback()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            TSNE_BH(
                early_exaggeration_iter=5,
                n_iter=5,
                callbacks=cb,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertEqual(cb.start_calls, 2)
        self.assertGreater(cb.iter_calls, 0)


if __name__ == "__main__":
    unittest.main()
