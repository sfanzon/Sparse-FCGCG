"""Numerical correctness and scaling-protocol regression tests."""

import unittest

import numpy as np

import experiment_scaling
import run_one
from algorithms import (
    coefficient_tolerance,
    fc_gcg,
    lasso_kkt_residual,
    soft_threshold,
)
from scaling_protocol import (
    METHODS,
    PROBLEM_SIZES,
    PROTOCOL_VERSION,
    make_instance,
    safe_lipschitz,
    solver_arguments,
    validate_results,
)


def independent_kkt_residual(A, b, x, lam):
    """Independent implementation used to check solver-reported metadata."""
    gradient = A.T @ (A @ x - b)
    threshold = coefficient_tolerance(x)
    values = []
    for xi, gi in zip(x, gradient):
        if abs(xi) > threshold:
            values.append(abs(gi + lam * np.sign(xi)))
        else:
            values.append(max(abs(gi) - lam, 0.0))
    return max(values, default=0.0)


class FCGCGCorrectnessTests(unittest.TestCase):
    def test_soft_threshold_and_identity_lasso(self):
        b = np.array([3.0, -2.0, 0.25, -0.1])
        lam = 0.5
        expected = np.array([2.5, -1.5, 0.0, 0.0])
        np.testing.assert_allclose(soft_threshold(b, lam), expected)

        A = np.eye(b.size)
        x, trace = fc_gcg(A, b, lam, n_iter=20)
        self.assertTrue(trace.converged)
        self.assertEqual(trace.status, "converged")
        np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)
        self.assertLessEqual(trace.kkt_residual, trace.kkt_tolerance)

    def test_scale_aware_stopping(self):
        A = np.eye(4)
        b = np.array([3.0, -2.0, 0.25, -0.1])
        lam = 0.5
        expected = soft_threshold(b, lam)

        classifications = []
        for scale in (1e-12, 1e12):
            x, trace = fc_gcg(
                np.sqrt(scale) * A,
                np.sqrt(scale) * b,
                scale * lam,
                n_iter=20,
            )
            classifications.append((trace.status, trace.converged))
            np.testing.assert_allclose(x, expected, rtol=1e-10, atol=1e-10)
            self.assertLessEqual(trace.kkt_residual, trace.kkt_tolerance)
            problem_scale = max(
                scale * lam,
                np.max(np.abs((np.sqrt(scale) * A).T @ (np.sqrt(scale) * b))),
            )
            self.assertLessEqual(
                trace.kkt_residual / problem_scale,
                trace.kkt_tolerance / problem_scale,
            )
        self.assertEqual(classifications[0], classifications[1])

    def test_reported_kkt_matches_independent_check(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((40, 80)) / np.sqrt(40)
        x_true = np.zeros(80)
        x_true[[3, 17, 51, 72]] = [2.0, -1.5, 1.0, -0.75]
        b = A @ x_true + 0.01 * rng.standard_normal(40)
        lam = 0.1 * np.max(np.abs(A.T @ b))

        x, trace = fc_gcg(A, b, lam, n_iter=100)
        checked = independent_kkt_residual(A, b, x, lam)
        helper_value = lasso_kkt_residual(A, b, x, lam)
        self.assertTrue(trace.converged)
        self.assertAlmostEqual(trace.kkt_residual, helper_value, places=14)
        self.assertAlmostEqual(trace.kkt_residual, checked, places=14)
        self.assertLessEqual(checked, trace.kkt_tolerance)

    def test_fully_corrective_objective_is_monotone_within_roundoff(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((50, 120)) / np.sqrt(50)
        x_true = np.zeros(120)
        x_true[[2, 19, 77, 101]] = [3.0, -2.0, 1.0, 0.5]
        b = A @ x_true + 0.01 * rng.standard_normal(50)
        lam = 0.08 * np.max(np.abs(A.T @ b))

        _, trace = fc_gcg(A, b, lam, n_iter=100)
        self.assertTrue(trace.converged)
        objectives = np.asarray(trace.obj)
        dimension = max(A.shape)
        d_eps = dimension * np.finfo(float).eps
        gamma_d = d_eps / (1.0 - d_eps)
        for previous, current in zip(objectives[:-1], objectives[1:]):
            allowance = gamma_d * max(1.0, abs(previous), abs(current))
            self.assertLessEqual(current, previous + allowance)

    def test_outer_and_inner_iteration_exhaustion_are_explicit(self):
        A = np.eye(3)
        b = np.array([3.0, -2.0, 1.0])
        lam = 0.25

        _, outer = fc_gcg(A, b, lam, n_iter=0)
        self.assertFalse(outer.converged)
        self.assertEqual(outer.status, "max_iter")
        self.assertGreater(outer.kkt_residual, outer.kkt_tolerance)
        self.assertNotIn("certif", outer.message.lower())

        _, inner = fc_gcg(A, b, lam, n_iter=5, inner_max_iter=0)
        self.assertFalse(inner.converged)
        self.assertEqual(inner.status, "inner_max_iter")
        self.assertEqual(inner.inner_status, "max_iter")
        self.assertFalse(inner.inner_converged)
        self.assertGreater(
            inner.inner_kkt_residual, inner.inner_kkt_tolerance
        )

    def test_floating_point_overflow_reports_numerical_error(self):
        with np.errstate(over="ignore", invalid="ignore"):
            _, trace = fc_gcg(
                np.array([[1e308]]),
                np.array([1e308]),
                1.0,
                n_iter=5,
            )
        self.assertFalse(trace.converged)
        self.assertEqual(trace.status, "numerical_error")


class ScalingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A_first, cls.b_first, cls.lam_first = make_instance(2_000)
        cls.A_second, cls.b_second, cls.lam_second = make_instance(2_000)

    def test_batch_and_resumable_protocols_are_identical(self):
        for method in ("ref",) + METHODS:
            self.assertEqual(
                experiment_scaling.batch_task_definition(2_000, method),
                run_one.resumable_task_definition(2_000, method),
            )
        self.assertIs(experiment_scaling.make_instance, run_one.make_instance)
        self.assertIs(experiment_scaling.build_reference, run_one.build_reference)
        self.assertIs(experiment_scaling.run_method, run_one.run_method)
        np.testing.assert_array_equal(self.A_first, self.A_second)
        np.testing.assert_array_equal(self.b_first, self.b_second)
        self.assertEqual(self.lam_first, self.lam_second)
        self.assertAlmostEqual(self.lam_first, 0.8134705898924287, places=14)
        # The audited 50-step power estimate was 12.4011535252 and unsafe.
        self.assertGreater(safe_lipschitz(self.A_first), 12.40115352521681)

        meta = {"tau": 1.0, "L": 2.0}
        for method in METHODS:
            batch_solver, batch_kwargs = solver_arguments(
                method, self.A_first, self.b_first, self.lam_first, meta
            )
            resumable_solver, resumable_kwargs = solver_arguments(
                method, self.A_first, self.b_first, self.lam_first, meta
            )
            self.assertIs(batch_solver, resumable_solver)
            self.assertEqual(batch_kwargs.keys(), resumable_kwargs.keys())
            for key in batch_kwargs:
                if isinstance(batch_kwargs[key], np.ndarray):
                    self.assertIs(batch_kwargs[key], resumable_kwargs[key])
                else:
                    self.assertEqual(batch_kwargs[key], resumable_kwargs[key])

    def test_lipschitz_bound_is_at_least_spectral_norm_squared(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        spectral_norm_squared = np.linalg.norm(A, 2) ** 2
        bound = safe_lipschitz(A)
        self.assertGreaterEqual(bound, spectral_norm_squared)

    def test_canonical_validation_rejects_legacy_results(self):
        results = {}
        for n in PROBLEM_SIZES:
            row = {
                "meta": {
                    "J_star": 1.0,
                    "tau": 1.0,
                    "L": 1.0,
                    "lam": 1.0,
                }
            }
            row.update({
                method: {"t": 1.0, "reached": True}
                for method in METHODS
            })
            results[str(n)] = row

        with self.assertRaisesRegex(ValueError, "protocol version None"):
            validate_results(results)

        for row in results.values():
            row["meta"]["protocol_version"] = PROTOCOL_VERSION
        self.assertIs(validate_results(results), results)


if __name__ == "__main__":
    unittest.main()
