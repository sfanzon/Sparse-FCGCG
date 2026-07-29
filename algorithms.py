"""
algorithms.py -- Four solvers for sparse regression, side by side.

Companion demo for:
    Bredies, Carioni, Fanzon, Walter (2024). "Asymptotic linear convergence
    of Fully-Corrective Generalized Conditional Gradient methods".
    Mathematical Programming 205:135-202.
    https://doi.org/10.1007/s10107-023-01975-z

The paper works in infinite-dimensional Banach spaces (optimization over
measures). This demo specialises everything to the cleanest possible finite-
dimensional instance -- the lasso --

        min_x  J(x) = 0.5 * ||A x - b||^2  +  lam * ||x||_1,        (P)

where the geometric picture of the paper becomes elementary:

  * The regularizer's unit ball {||x||_1 <= 1} is the cross-polytope.
  * Its EXTREMAL POINTS are exactly the 2n signed coordinate vectors +-e_i.
  * "Sparse solution" literally means "conic combination of few extremal
    points" -- which is the design principle of the FC-GCG iterates.

Solvers implemented (all track objective value, support size, wall time):

  ista(...)         proximal gradient descent (the "gradient descent for
                    sparse problems" baseline)
  fista(...)        its accelerated variant (Beck & Teboulle 2009)
  frank_wolfe(...)  vanilla FW on the constraint set {||x||_1 <= tau}
  fc_gcg(...)       the paper's algorithm, specialised to (P):
                    insertion step + fully-corrective step

Only numpy is required.
"""

from dataclasses import dataclass
import time
import numpy as np


# ----------------------------------------------------------------- helpers --

DEFAULT_KKT_RTOL = float(np.sqrt(np.finfo(float).eps))
# Tighter tolerance for solutions used as numerical objective references.
REFERENCE_KKT_RTOL = 1e-10


def objective(A, b, x, lam):
    """Penalised lasso objective J(x)."""
    r = A @ x - b
    return 0.5 * r @ r + lam * np.abs(x).sum()


def soft_threshold(v, t):
    """Proximal operator of t*||.||_1 (elementwise soft-thresholding)."""
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def coefficient_tolerance(x):
    """Machine-scale threshold used consistently for pruning and KKT activity."""
    scale = max(1.0, float(np.max(np.abs(x), initial=0.0)))
    return np.finfo(float).eps * scale


def lasso_kkt_residual(A, b, x, lam, gradient=None):
    """
    Infinity norm of the complete lasso KKT residual.

    Active coordinates use |g_i + lam*sign(x_i)|. Coordinates pruned at the
    solver's machine-scale coefficient threshold use max(|g_i| - lam, 0).
    """
    g = A.T @ (A @ x - b) if gradient is None else np.asarray(gradient)
    if g.shape != x.shape:
        raise ValueError("gradient must have the same shape as x")
    active = np.abs(x) > coefficient_tolerance(x)
    residual = np.empty_like(g, dtype=float)
    residual[active] = np.abs(g[active] + lam * np.sign(x[active]))
    residual[~active] = np.maximum(np.abs(g[~active]) - lam, 0.0)
    return float(np.max(residual, initial=0.0))


def lasso_kkt_scale(A, b, lam):
    """Fixed problem scale for absolute-plus-relative KKT tolerances."""
    dual_at_zero = A.T @ b
    if not np.all(np.isfinite(dual_at_zero)):
        return np.inf
    return max(float(lam), float(np.max(np.abs(dual_at_zero), initial=0.0)))


def lasso_kkt_tolerance(A, b, lam, atol=0.0, rtol=DEFAULT_KKT_RTOL):
    """Return atol + rtol*S, S=max(lam, ||A^T b||_inf)."""
    scale = lasso_kkt_scale(A, b, lam)
    if not np.isfinite(scale):
        return np.inf
    return float(atol + rtol * scale)


class Trace:
    """Per-iteration history plus final convergence metadata."""

    def __init__(self):
        self.obj, self.nnz, self.t = [], [], []
        self._t0 = time.perf_counter()
        self.status = "max_iter"
        self.converged = False
        self.iterations = 0
        self.kkt_residual = np.inf
        self.kkt_tolerance = np.nan
        self.objective = np.nan
        self.support_size = 0
        self.message = ""
        self.inner_status = "not_run"
        self.inner_converged = False
        self.inner_iterations = 0
        self.inner_kkt_residual = np.nan
        self.inner_kkt_tolerance = np.nan

    def log(self, obj, nnz):
        self.obj.append(obj)
        self.nnz.append(nnz)
        self.t.append(time.perf_counter() - self._t0)


@dataclass
class RestrictedSolveResult:
    """Result of the fully-corrective lasso solve on the active columns."""

    solution: np.ndarray
    iterations: int
    status: str
    converged: bool
    kkt_residual: float
    kkt_tolerance: float


# ------------------------------------------------------- proximal gradient --

def ista(A, b, lam, n_iter=5000, x0=None, trace_every=1, L=None):
    """Proximal gradient descent with fixed step 1/L, L = ||A||_2^2."""
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    if L is None:
        L = np.linalg.norm(A, 2) ** 2
    tr = Trace()
    for k in range(n_iter):
        g = A.T @ (A @ x - b)                     # full gradient: O(mn)
        x = soft_threshold(x - g / L, lam / L)
        if k % trace_every == 0:
            tr.log(objective(A, b, x, lam), int((x != 0).sum()))
    return x, tr


def fista(A, b, lam, n_iter=5000, x0=None, trace_every=1, L=None):
    """Accelerated proximal gradient (FISTA, Beck & Teboulle 2009)."""
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    y, t_mom = x.copy(), 1.0
    if L is None:
        L = np.linalg.norm(A, 2) ** 2
    tr = Trace()
    for k in range(n_iter):
        g = A.T @ (A @ y - b)
        x_new = soft_threshold(y - g / L, lam / L)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_mom ** 2))
        y = x_new + ((t_mom - 1) / t_new) * (x_new - x)
        x, t_mom = x_new, t_new
        if k % trace_every == 0:
            tr.log(objective(A, b, x, lam), int((x != 0).sum()))
    return x, tr


# ------------------------------------------------------------- Frank-Wolfe --

def frank_wolfe(A, b, tau, n_iter=5000, trace_every=1, lam_for_obj=0.0):
    """
    Vanilla Frank-Wolfe / conditional gradient on the CONSTRAINED problem

            min_x 0.5||Ax - b||^2   s.t.  ||x||_1 <= tau.

    Classical ingredients, unchanged since Frank & Wolfe (1956):

      * Linear minimization oracle. Minimising a linear function over the
        l1 ball is attained at an EXTREMAL POINT (a vertex +-tau*e_i of the
        cross-polytope) -- the same "key lemma" the paper uses in Banach
        spaces. Concretely: i* = argmax_i |g_i|, vertex = -tau*sign(g_i*)e_i*.
      * Step size gamma_k = 2/(k+2), the standard open-loop schedule.

    Guarantees O(1/k) objective decay -- and iterates that are DENSE
    convex combinations of all vertices visited so far, with the notorious
    zig-zagging as the solution is approached.

    For comparability with the penalised solvers, the logged objective is
    0.5||Ax-b||^2 + lam_for_obj*||x||_1 (with ||x||_1 ~= tau at optimality
    this differs from the constrained objective by ~ a constant).
    """
    m, n = A.shape
    x = np.zeros(n)
    tr = Trace()
    for k in range(n_iter):
        g = A.T @ (A @ x - b)                     # full gradient: O(mn)
        i = int(np.argmax(np.abs(g)))             # LMO over extremal points
        v = np.zeros(n)
        v[i] = -tau * np.sign(g[i])
        gamma = 2.0 / (k + 2.0)
        x = (1 - gamma) * x + gamma * v           # convex combination
        if k % trace_every == 0:
            tr.log(objective(A, b, x, lam_for_obj), int((x != 0).sum()))
    return x, tr


# ------------------------------------------------------------------ FC-GCG --

def _lasso_on_support(
    AS,
    b,
    lam,
    z0,
    atol=0.0,
    rtol=DEFAULT_KKT_RTOL,
    max_iter=10000,
):
    """
    Fully-corrective inner problem on the current active columns.

    Cyclic coordinate descent is cheap because s = #atoms is tiny. Convergence
    is decided by the restricted lasso KKT residual, not coefficient change.
    """
    s = AS.shape[1]
    z = z0.copy()
    col_sq = (AS ** 2).sum(axis=0)                # ||a_j||^2
    r = b - AS @ z                                # residual
    kkt_tolerance = lasso_kkt_tolerance(AS, b, lam, atol=atol, rtol=rtol)
    if not np.isfinite(kkt_tolerance):
        return RestrictedSolveResult(
            z, 0, "numerical_error", False, np.inf, kkt_tolerance
        )
    initial_residual = lasso_kkt_residual(AS, b, z, lam)
    if not np.isfinite(initial_residual):
        return RestrictedSolveResult(
            z, 0, "numerical_error", False, np.inf, kkt_tolerance
        )
    if initial_residual <= kkt_tolerance:
        return RestrictedSolveResult(
            z, 0, "converged", True, initial_residual, kkt_tolerance
        )

    for iteration in range(1, max_iter + 1):
        for j in range(s):
            if col_sq[j] == 0.0:
                continue
            zj_old = z[j]
            rho = AS[:, j] @ r + col_sq[j] * zj_old
            z[j] = soft_threshold(rho, lam) / col_sq[j]
            dz = z[j] - zj_old
            if dz != 0.0:
                r -= AS[:, j] * dz
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(r)):
            return RestrictedSolveResult(
                z, iteration, "numerical_error", False, np.inf, kkt_tolerance
            )
        kkt_residual = lasso_kkt_residual(AS, b, z, lam)
        if not np.isfinite(kkt_residual):
            return RestrictedSolveResult(
                z, iteration, "numerical_error", False, np.inf, kkt_tolerance
            )
        if kkt_residual <= kkt_tolerance:
            return RestrictedSolveResult(
                z, iteration, "converged", True, kkt_residual, kkt_tolerance
            )

    final_residual = lasso_kkt_residual(AS, b, z, lam)
    return RestrictedSolveResult(
        z, max_iter, "max_iter", False, final_residual, kkt_tolerance
    )


def fc_gcg(
    A,
    b,
    lam,
    n_iter=200,
    atol=0.0,
    rtol=DEFAULT_KKT_RTOL,
    trace_every=1,
    inner_max_iter=10000,
):
    """
    Fully-Corrective Generalized Conditional Gradient, specialised to the
    lasso. Direct translation of the paper's two-step scheme:

    INSERTION STEP (one linear problem).
        Compute the dual variable p = A^T(b - Ax) and find the extremal
        point of the regularizer's unit ball most correlated with it:
        i* = argmax_i |p_i|. The algorithm stops only when the complete lasso
        KKT residual is below a scale-aware numerical tolerance. Otherwise add
        the signed atom sign(p_i*) e_i* to the active set A_k.

    FULLY-CORRECTIVE STEP (one small convex problem).
        Re-solve (P) restricted to cone(A_k) -- here simply the lasso on
        the active columns, a problem of dimension #atoms << n -- and
        drop any atom whose coefficient is zero.

    The iterate is therefore a conic combination of few extremal points
    AT EVERY ITERATION: sparsity by construction, not as a limit. Under
    the paper's nondegeneracy assumptions the method converges (locally)
    LINEARLY; in well-conditioned instances it typically terminates in
    about (#atoms of the solution) + a few iterations.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.size:
        raise ValueError("A must be a matrix and b a matching vector")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite values")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and non-negative")
    if n_iter < 0 or inner_max_iter < 0:
        raise ValueError("iteration limits must be non-negative")
    if trace_every < 1:
        raise ValueError("trace_every must be at least one")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")

    m, n = A.shape
    support = []                                  # indices of active atoms
    z = np.zeros(0)                               # their coefficients
    tr = Trace()
    kkt_tolerance = lasso_kkt_tolerance(A, b, lam, atol=atol, rtol=rtol)
    previous_objective = objective(A, b, np.zeros(n), lam)
    last_inner = None

    def finish(status, converged, iterations, x, residual, message):
        tr.status = status
        tr.converged = converged
        tr.iterations = iterations
        tr.kkt_residual = float(residual)
        tr.kkt_tolerance = kkt_tolerance
        tr.objective = float(objective(A, b, x, lam))
        tr.support_size = int(np.count_nonzero(x))
        tr.message = message
        if last_inner is not None:
            tr.inner_status = last_inner.status
            tr.inner_converged = last_inner.converged
            tr.inner_iterations = last_inner.iterations
            tr.inner_kkt_residual = last_inner.kkt_residual
            tr.inner_kkt_tolerance = last_inner.kkt_tolerance
        return x, tr

    if not np.isfinite(kkt_tolerance) or not np.isfinite(previous_objective):
        return finish(
            "numerical_error", False, 0, np.zeros(n), np.inf,
            "non-finite initial objective or KKT scale",
        )

    for iteration in range(n_iter + 1):
        x = np.zeros(n)
        x[support] = z
        current_objective = objective(A, b, x, lam)
        g = A.T @ (A @ x - b)
        if not np.isfinite(current_objective) or not np.all(np.isfinite(g)):
            return finish(
                "numerical_error", False, iteration, x, np.inf,
                "non-finite objective or gradient",
            )
        kkt_residual = lasso_kkt_residual(A, b, x, lam, gradient=g)
        if not np.isfinite(kkt_residual):
            return finish(
                "numerical_error", False, iteration, x, np.inf,
                "non-finite KKT residual",
            )
        if kkt_residual <= kkt_tolerance:
            return finish(
                "converged", True, iteration, x, kkt_residual,
                "full lasso KKT residual is within tolerance",
            )
        if iteration == n_iter:
            return finish(
                "max_iter", False, iteration, x, kkt_residual,
                "outer iteration limit reached before KKT convergence",
            )

        p = -g                                      # A^T(b - Ax)
        i = int(np.argmax(np.abs(p)))
        if i not in support:                      # insertion step
            support.append(i)
            z = np.append(z, 0.0)

        AS = A[:, support]                        # fully-corrective step:
        last_inner = _lasso_on_support(
            AS, b, lam, z, atol=atol, rtol=rtol, max_iter=inner_max_iter
        )
        z = last_inner.solution
        if last_inner.status == "numerical_error":
            x = np.zeros(n)
            x[support] = z
            return finish(
                "numerical_error", False, iteration + 1, x, np.inf,
                "restricted lasso solve produced non-finite values",
            )
        if not last_inner.converged:
            x = np.zeros(n)
            x[support] = z
            residual = lasso_kkt_residual(A, b, x, lam)
            return finish(
                "inner_max_iter", False, iteration + 1, x, residual,
                "restricted lasso iteration limit reached",
            )

        prune_tol = coefficient_tolerance(z)
        keep = np.abs(z) > prune_tol               # drop numerical zero atoms
        z[~keep] = 0.0
        support = [s_ for s_, kp in zip(support, keep) if kp]
        z = z[keep]

        x_new = np.zeros(n)
        x_new[support] = z
        new_objective = objective(A, b, x_new, lam)
        if not np.isfinite(new_objective):
            return finish(
                "numerical_error", False, iteration + 1, x_new, np.inf,
                "fully-corrective step produced a non-finite objective",
            )
        # Standard dot-product roundoff bound gamma_d = d*eps/(1-d*eps).
        # This permits dimension-scaled floating-point noise, not material
        # increases in the fully-corrective objective.
        d = max(A.shape)
        d_eps = d * np.finfo(float).eps
        gamma_d = d_eps / (1.0 - d_eps) if d_eps < 1.0 else np.inf
        monotonicity_allowance = gamma_d * max(
            1.0, abs(previous_objective), abs(new_objective)
        )
        if new_objective > previous_objective + monotonicity_allowance:
            residual = lasso_kkt_residual(A, b, x_new, lam)
            return finish(
                "numerical_error", False, iteration + 1, x_new, residual,
                "fully-corrective objective increased beyond roundoff",
            )

        if iteration % trace_every == 0:
            tr.log(new_objective, len(support))
        previous_objective = new_objective

    raise RuntimeError("unreachable FC-GCG termination state")
