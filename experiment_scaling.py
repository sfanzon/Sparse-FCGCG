"""
experiment_scaling.py -- Wall-clock time to solve the lasso as the
dimension n grows: FC-GCG vs Frank-Wolfe vs (F)ISTA.

Protocol. For n in {2 000, 20 000, 200 000} (m = 300, 15-sparse truth):
measure wall time until the objective is within REL_TOL of the optimum
J* (computed by running FC-GCG to its exact KKT certificate). Iteration
caps: any method that fails to reach the tolerance within MAX_ITER
iterations is reported at its capped time with an open marker.

The point being illustrated: all four methods pay the same O(mn) for one
gradient/dual evaluation, so the game is entirely about HOW MANY
iterations you need. FC-GCG needs roughly (#atoms of the solution) of
them, independent of n; first-order methods need hundreds to thousands,
with constants that degrade with conditioning.

Run:  python3 experiment_scaling.py       (takes a few minutes)
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import ista, fista, frank_wolfe, fc_gcg, objective

rng = np.random.default_rng(1)

M, K_TRUE = 300, 15
NS = [2_000, 20_000, 200_000]
REL_TOL = 1e-8
MAX_ITER = 2000


def lipschitz(A, iters=50):
    """||A||_2^2 by power iteration (SVD would be prohibitive for large n)."""
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = A.T @ (A @ v)
        v /= np.linalg.norm(v)
    return float(v @ (A.T @ (A @ v)))


def time_to_tol(solver, J_target, **kw):
    """Run solver, return (time_to_target, reached?)."""
    t0 = time.perf_counter()
    _, tr = solver(**kw)
    obj = np.array(tr.obj)
    hit = np.nonzero(obj <= J_target)[0]
    if len(hit):
        return tr.t[hit[0]], True
    return time.perf_counter() - t0, False


results = {name: [] for name in ["FC-GCG", "Frank-Wolfe", "ISTA", "FISTA"]}
reached = {name: [] for name in results}

for n in NS:
    print(f"\n--- n = {n} ---")
    A = rng.standard_normal((M, n)) / np.sqrt(M)
    x_true = np.zeros(n)
    x_true[rng.choice(n, K_TRUE, replace=False)] = rng.standard_normal(K_TRUE) * 3
    b = A @ x_true + 0.02 * rng.standard_normal(M)
    lam = 0.1 * np.abs(A.T @ b).max()
    L = lipschitz(A)

    x_star, tr_star = fc_gcg(A, b, lam, n_iter=500, kkt_tol=1e-12)
    J_star = objective(A, b, x_star, lam)
    J_target = J_star + REL_TOL * max(1.0, abs(J_star))
    tau = np.abs(x_star).sum()

    runs = {
        "FC-GCG":      (fc_gcg,      dict(A=A, b=b, lam=lam, n_iter=500)),
        "Frank-Wolfe": (frank_wolfe, dict(A=A, b=b, tau=tau, n_iter=MAX_ITER, trace_every=10,
                                          lam_for_obj=lam)),
        "ISTA":        (ista,        dict(A=A, b=b, lam=lam, n_iter=MAX_ITER, trace_every=10,
                                          L=L)),
        "FISTA":       (fista,       dict(A=A, b=b, lam=lam, n_iter=MAX_ITER, trace_every=10,
                                          L=L)),
    }
    for name, (solver, kw) in runs.items():
        t, ok = time_to_tol(solver, J_target, **kw)
        results[name].append(t)
        reached[name].append(ok)
        print(f"  {name:12s} {t:8.2f}s   "
              f"{'reached tol' if ok else f'NOT reached in {MAX_ITER} iters'}")

# -------------------------------------------------------------------- plot --
STYLE = {"FC-GCG": "#C8102E", "Frank-Wolfe": "#1f77b4",
         "ISTA": "#7f7f7f", "FISTA": "#2ca02c"}
fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
for name in results:
    t = np.array(results[name])
    ok = np.array(reached[name])
    ax.loglog(NS, t, "-o", color=STYLE[name], lw=2, label=name)
    if (~ok).any():   # open markers where the cap was hit
        ax.loglog(np.array(NS)[~ok], t[~ok], "o", mfc="white",
                  color=STYLE[name], ms=9)
ax.set_xlabel("problem dimension $n$   (m = 300 fixed)")
ax.set_ylabel(f"wall time to rel. gap ${REL_TOL:g}$  [s]")
ax.set_title("Time to solve the lasso vs dimension\n"
             "(open markers: tolerance NOT reached within iteration cap)")
ax.legend(frameon=False)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("figures/scaling.png")
print("\nfigure written to figures/scaling.png")
