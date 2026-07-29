"""
experiment_convergence.py -- Convergence & sparsity: FC-GCG vs FW vs (F)ISTA.

Instance: lasso with A (300 x 5000) Gaussian, 12-sparse ground truth,
mild noise, lambda = 0.1 * ||A^T b||_inf.

Produces:
    figures/convergence.png    objective suboptimality vs iteration (log y)
    figures/sparsity.png       support size of the iterate vs iteration

Run:  python3 experiment_convergence.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import ista, fista, frank_wolfe, fc_gcg, objective

rng = np.random.default_rng(0)

# ------------------------------------------------------------ the instance --
m, n, k_true = 300, 5000, 12
A = rng.standard_normal((m, n)) / np.sqrt(m)
x_true = np.zeros(n)
supp = rng.choice(n, k_true, replace=False)
x_true[supp] = rng.standard_normal(k_true) * 3
b = A @ x_true + 0.02 * rng.standard_normal(m)
lam = 0.1 * np.abs(A.T @ b).max()

# --------------------------------------------------------------- reference --
# Numerical reference: FC-GCG must satisfy its full scale-aware KKT check.
x_star, tr_star = fc_gcg(A, b, lam, n_iter=500)
if not tr_star.converged:
    raise RuntimeError(f"reference FC-GCG failed: {tr_star.status}")
J_star = objective(A, b, x_star, lam)
tau = np.abs(x_star).sum()          # matched l1 budget for vanilla FW

# ------------------------------------------------------------------- solve --
N = 2000
_, tr_ista  = ista(A, b, lam, n_iter=N)
_, tr_fista = fista(A, b, lam, n_iter=N)
_, tr_fw    = frank_wolfe(A, b, tau, n_iter=N, lam_for_obj=lam)
_, tr_gcg   = fc_gcg(A, b, lam, n_iter=100)

FLOOR = 1e-14
def gap(tr):
    return np.maximum(np.array(tr.obj) - J_star, FLOOR)

# ----------------------------------------------------------------- figures --
STYLE = {
    "FC-GCG (the paper)":      dict(color="#C8102E", lw=2.5),
    "Vanilla Frank-Wolfe":     dict(color="#1f77b4", lw=1.8),
    "ISTA (prox. gradient)":   dict(color="#7f7f7f", lw=1.8),
    "FISTA (accelerated)":     dict(color="#2ca02c", lw=1.8),
}
runs = {
    "FC-GCG (the paper)": tr_gcg,
    "Vanilla Frank-Wolfe": tr_fw,
    "ISTA (prox. gradient)": tr_ista,
    "FISTA (accelerated)": tr_fista,
}

fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
for name, tr in runs.items():
    ax.semilogy(gap(tr), label=name, **STYLE[name])
ax.set_xlim(0, 300)
ax.set_ylim(FLOOR / 10, None)
ax.set_xlabel("iteration")
ax.set_ylabel(r"objective suboptimality  $J(x_k) - J^\ast$")
ax.set_title(f"Lasso, $A \\in \\mathbb{{R}}^{{{m} \\times {n}}}$, "
             f"{k_true}-sparse truth")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/convergence.png")

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=170)
for name, tr in runs.items():
    ax.plot(tr.nnz, label=name, **STYLE[name])
ax.axhline(k_true, color="k", ls=":", lw=1)
ax.text(295, k_true + 6, "true sparsity", ha="right", fontsize=9)
ax.set_xlim(0, 300)
ax.set_xlabel("iteration")
ax.set_ylabel("support size of iterate")
ax.set_title("Sparsity along the way: FC-GCG iterates are sparse by design")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("figures/sparsity.png")

# ----------------------------------------------------------------- summary --
print(f"lambda = {lam:.4f},  J* = {J_star:.8f},  ||x*||_0 = {(x_star!=0).sum()}")
print(f"FC-GCG stopped after {tr_gcg.iterations} iterations "
      f"({tr_gcg.status}; KKT {tr_gcg.kkt_residual:.3e} <= "
      f"{tr_gcg.kkt_tolerance:.3e})")
for name, tr in runs.items():
    print(f"{name:26s} final gap {gap(tr)[-1]:.2e}   "
          f"final nnz {tr.nnz[-1]:5d}   iters {len(tr.obj)}")
