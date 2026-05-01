"""Tests for the learnable degrees-of-freedom feature."""
import logging
import unittest
from functools import partial

import numpy as np
from sklearn import datasets

import openTSNE
from openTSNE import affinity
from openTSNE.tsne import (
    GradientResult,
    OptimizationStats,
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

    def test_optimize_returns_three_tuple(self):
        # The optimizer's outer return is (error, embedding, optimization_stats).
        emb = TSNE_BH(early_exaggeration_iter=5, n_iter=5).fit(self.x)
        self.assertTrue(hasattr(emb, "kl_divergence"))
        self.assertTrue(hasattr(emb, "optimization_stats"))
        self.assertIsInstance(emb.optimization_stats, OptimizationStats)


class TestDofAutoLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_bh_auto_actually_moves_dof(self):
        # With dof="auto" on the BH path, dof should change over iterations.
        emb = TSNE_BH(
            dof="auto",
            initial_dof=1.0,
            early_exaggeration_iter=0,
            n_iter=30,
        ).fit(self.x)
        alphas = emb.optimization_stats.alphas
        self.assertGreater(len(alphas), 0)
        # First recorded alpha is the initial; final should differ.
        self.assertNotEqual(alphas[0], alphas[-1])

    def test_fft_auto_warns_and_keeps_dof_fixed(self):
        # FFT path cannot learn dof; we must warn and dof must remain fixed.
        with self.assertLogs("openTSNE.tsne", level="WARNING") as cm:
            emb = TSNE_FFT(
                dof="auto",
                initial_dof=1.0,
                early_exaggeration_iter=0,
                n_iter=20,
            ).fit(self.x)
        self.assertTrue(
            any("dof='auto'" in msg or "Barnes-Hut" in msg for msg in cm.output),
            f"Expected dof-auto warning, got: {cm.output}",
        )
        alphas = emb.optimization_stats.alphas
        self.assertTrue(all(a == 1.0 for a in alphas))


if __name__ == "__main__":
    unittest.main()
