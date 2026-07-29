"""Canonical deterministic protocol for the dimension-scaling experiment."""

from dataclasses import dataclass
import time

import numpy as np

from algorithms import (
    DEFAULT_KKT_RTOL,
    fc_gcg,
    fista,
    frank_wolfe,
    ista,
    objective,
)


M = 300
K_TRUE = 15
PROTOCOL_VERSION = 2
PROBLEM_SIZES = (2_000, 20_000, 200_000)
REL_TOL = 1e-8
ITERATION_CAPS = {2_000: 2_000, 20_000: 2_000, 200_000: 1_200}
METHODS = ("gcg", "fw", "ista", "fista")
REFERENCE_MAX_ITER = 500
GCG_MAX_ITER = 500
TRACE_EVERY = 10
KKT_ATOL = 0.0
KKT_RTOL = DEFAULT_KKT_RTOL
# The reference KKT residual is required to be 100 times smaller than the
# reported relative objective-gap target. This keeps reference error from
# deciding whether a method reaches REL_TOL.
REFERENCE_KKT_RTOL = REL_TOL / 100.0


@dataclass(frozen=True)
class ScalingTask:
    """Serializable choices defining one timing task."""

    n: int
    method: str
    seed: int
    iteration_cap: int
    relative_tolerance: float
    kkt_atol: float
    kkt_rtol: float
    trace_every: int


def seed_for_dimension(n):
    """The audited resumable protocol derives each instance seed from n."""
    return int(n)


def iteration_cap(n):
    try:
        return ITERATION_CAPS[int(n)]
    except KeyError as exc:
        raise ValueError(f"unsupported problem size: {n}") from exc


def task_definition(n, method):
    """Return every protocol choice needed for one resumable/batch task."""
    n = int(n)
    if method not in ("ref",) + METHODS:
        raise ValueError(f"unsupported method: {method}")
    return ScalingTask(
        n=n,
        method=method,
        seed=seed_for_dimension(n),
        iteration_cap=iteration_cap(n),
        relative_tolerance=REL_TOL,
        kkt_atol=KKT_ATOL,
        kkt_rtol=KKT_RTOL,
        trace_every=TRACE_EVERY,
    )


def make_instance(n):
    """Generate the canonical deterministic synthetic lasso instance."""
    task = task_definition(n, "ref")
    rng = np.random.default_rng(task.seed)
    A = rng.standard_normal((M, task.n)) / np.sqrt(M)
    x_true = np.zeros(task.n)
    # Preserve the audited run_one.py random-draw order: Python evaluates the
    # assignment value before the indexed target in
    # x_true[rng.choice(...)] = rng.standard_normal(...).
    values = rng.standard_normal(K_TRUE) * 3
    support = rng.choice(task.n, K_TRUE, replace=False)
    x_true[support] = values
    b = A @ x_true + 0.02 * rng.standard_normal(M)
    lam = 0.1 * np.max(np.abs(A.T @ b))
    return A, b, lam


def safe_lipschitz(A):
    """
    Compute ||A||_2^2 from the smaller symmetric Gram matrix.

    A and A.T have the same non-zero squared singular values as A @ A.T.
    Unlike a fixed-count power iteration, eigvalsh does not systematically
    underestimate the largest eigenvalue. nextafter rounds the returned value
    upward by one representable float for use as an ISTA/FISTA step bound.
    """
    gram = A @ A.T
    largest = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(largest) or largest < 0:
        raise FloatingPointError("failed to compute a finite Lipschitz bound")
    return float(np.nextafter(largest, np.inf))


def build_reference(A, b, lam):
    """Compute and validate the objective reference, l1 radius and step bound."""
    x_star, trace = fc_gcg(
        A,
        b,
        lam,
        n_iter=REFERENCE_MAX_ITER,
        atol=KKT_ATOL,
        rtol=REFERENCE_KKT_RTOL,
    )
    if not trace.converged:
        raise RuntimeError(
            f"reference FC-GCG did not converge: {trace.status} "
            f"(KKT {trace.kkt_residual:.3e} > {trace.kkt_tolerance:.3e})"
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "J_star": float(objective(A, b, x_star, lam)),
        "tau": float(np.abs(x_star).sum()),
        "L": safe_lipschitz(A),
        "lam": float(lam),
        "reference_status": trace.status,
        "reference_kkt_residual": float(trace.kkt_residual),
        "reference_kkt_tolerance": float(trace.kkt_tolerance),
    }


def objective_target(meta):
    """Canonical absolute-plus-relative objective target."""
    j_star = float(meta["J_star"])
    return j_star + REL_TOL * max(1.0, abs(j_star))


def solver_arguments(method, A, b, lam, meta):
    """Return the canonical solver and keyword arguments for one method."""
    cap = iteration_cap(A.shape[1])
    if method == "gcg":
        return fc_gcg, {
            "A": A,
            "b": b,
            "lam": lam,
            "n_iter": GCG_MAX_ITER,
            "atol": KKT_ATOL,
            "rtol": KKT_RTOL,
        }
    if method == "fw":
        return frank_wolfe, {
            "A": A,
            "b": b,
            "tau": meta["tau"],
            "n_iter": cap,
            "trace_every": TRACE_EVERY,
            "lam_for_obj": lam,
        }
    if method == "ista":
        return ista, {
            "A": A,
            "b": b,
            "lam": lam,
            "n_iter": cap,
            "trace_every": TRACE_EVERY,
            "L": meta["L"],
        }
    if method == "fista":
        return fista, {
            "A": A,
            "b": b,
            "lam": lam,
            "n_iter": cap,
            "trace_every": TRACE_EVERY,
            "L": meta["L"],
        }
    raise ValueError(f"unsupported method: {method}")


def run_method(method, A, b, lam, meta):
    """Run one canonical timing task and report time-to-target."""
    solver, kwargs = solver_arguments(method, A, b, lam, meta)
    target = objective_target(meta)
    started = time.perf_counter()
    _, trace = solver(**kwargs)
    objectives = np.asarray(trace.obj)
    hits = np.flatnonzero(objectives <= target)
    if hits.size:
        elapsed = float(trace.t[int(hits[0])])
        reached = True
    else:
        elapsed = float(time.perf_counter() - started)
        reached = False
    result = {"t": elapsed, "reached": reached}
    if method == "gcg":
        result.update({
            "status": trace.status,
            "kkt_residual": float(trace.kkt_residual),
            "kkt_tolerance": float(trace.kkt_tolerance),
        })
    return result


def validate_results(results):
    """Validate the JSON structure consumed by the scaling plot."""
    if not isinstance(results, dict) or not results:
        raise ValueError("scaling results must be a non-empty object")
    for n in PROBLEM_SIZES:
        row = results.get(str(n))
        if not isinstance(row, dict):
            raise ValueError(f"missing results for n={n}")
        meta = row.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"missing reference metadata for n={n}")
        # Legacy committed results are retained until the expensive benchmark
        # is deliberately regenerated. They may still be plotted, but are
        # never accepted as solver input by run_one.py. Any explicitly
        # versioned result must match the current canonical protocol.
        version = meta.get("protocol_version")
        if version is not None and version != PROTOCOL_VERSION:
            raise ValueError(
                f"results for n={n} use protocol version {version}, "
                f"expected {PROTOCOL_VERSION}"
            )
        for key in ("J_star", "tau", "L", "lam"):
            if key not in meta or not np.isfinite(meta[key]):
                raise ValueError(f"invalid {key} for n={n}")
        if meta["L"] <= 0 or meta["tau"] < 0 or meta["lam"] < 0:
            raise ValueError(f"invalid non-positive metadata for n={n}")
        for method in METHODS:
            value = row.get(method)
            if not isinstance(value, dict):
                raise ValueError(f"missing {method} result for n={n}")
            if not np.isfinite(value.get("t", np.nan)) or value["t"] < 0:
                raise ValueError(f"invalid {method} time for n={n}")
            if not isinstance(value.get("reached"), bool):
                raise ValueError(f"invalid {method} reached flag for n={n}")
    return results
