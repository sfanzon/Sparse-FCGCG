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

import time
import numpy as np


# ----------------------------------------------------------------- helpers --

def objective(A, b, x, lam):
    """Penalised lasso objective J(x)."""
    r = A @ x - b
    return 0.5 * r @ r + lam * np.abs(x).sum()


def soft_threshold(v, t):
    """Proximal operator of t*||.||_1 (elementwise soft-thresholding)."""
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class Trace:
    """Per-iteration record: objective, support size, elapsed wall time."""

    def __init__(self):
        self.obj, self.nnz, self.t = [], [], []
        self._t0 = time.perf_counter()

    def log(self, obj, nnz):
        self.obj.append(obj)
        self.nnz.append(nnz)
        self.t.append(time.perf_counter() - self._t0)


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

def _lasso_on_support(AS, b, lam, z0, tol=1e-12, max_iter=10000):
    """
    Fully-corrective inner problem: solve the lasso EXACTLY (to tol) but
    restricted to the current active columns AS (a small m x s matrix).
    Cyclic coordinate descent -- cheap because s = #atoms is tiny.
    """
    s = AS.shape[1]
    z = z0.copy()
    col_sq = (AS ** 2).sum(axis=0)                # ||a_j||^2
    r = b - AS @ z                                # residual
    for _ in range(max_iter):
        z_max_change = 0.0
        for j in range(s):
            if col_sq[j] == 0.0:
                continue
            zj_old = z[j]
            rho = AS[:, j] @ r + col_sq[j] * zj_old
            z[j] = soft_threshold(rho, lam) / col_sq[j]
            dz = z[j] - zj_old
            if dz != 0.0:
                r -= AS[:, j] * dz
                z_max_change = max(z_max_change, abs(dz))
        if z_max_change < tol:
            break
    return z


def fc_gcg(A, b, lam, n_iter=200, kkt_tol=1e-10, trace_every=1):
    """
    Fully-Corrective Generalized Conditional Gradient, specialised to the
    lasso. Direct translation of the paper's two-step scheme:

    INSERTION STEP (one linear problem).
        Compute the dual variable p = A^T(b - Ax) and find the extremal
        point of the regularizer's unit ball most correlated with it:
        i* = argmax_i |p_i|. If max_i |p_i| <= lam, the KKT conditions of
        (P) hold and we STOP -- the algorithm detects optimality exactly,
        it does not just drift towards it. Otherwise add the signed atom
        sign(p_i*) e_i* to the active set A_k.

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
    m, n = A.shape
    support = []                                  # indices of active atoms
    z = np.zeros(0)                               # their coefficients
    tr = Trace()
    for k in range(n_iter):
        x_res = b - (A[:, support] @ z if support else np.zeros(m))
        p = A.T @ x_res                           # dual variable: O(mn)
        i = int(np.argmax(np.abs(p)))

        if np.abs(p[i]) <= lam + kkt_tol:         # exact KKT certificate
            x = np.zeros(n); x[support] = z
            tr.log(objective(A, b, x, lam), len(support))
            break

        if i not in support:                      # insertion step
            support.append(i)
            z = np.append(z, 0.0)

        AS = A[:, support]                        # fully-corrective step:
        z = _lasso_on_support(AS, b, lam, z)      # small exact solve
        keep = z != 0.0                           # drop dead atoms
        support = [s_ for s_, kp in zip(support, keep) if kp]
        z = z[keep]

        if k % trace_every == 0:
            x = np.zeros(n); x[support] = z
            tr.log(objective(A, b, x, lam), len(support))

    x = np.zeros(n)
    x[support] = z
    return x, tr
