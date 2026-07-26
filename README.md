# Sparse optimization by extremal points: FC-GCG vs Frank-Wolfe vs gradient descent

A self-contained numerical companion to the paper:

> Bredies, K., Carioni, M., Fanzon, S., Walter, D. (2024). *Asymptotic linear convergence of fully-corrective generalized conditional gradient methods*. **Mathematical Programming**, 205:135–202. [doi:10.1007/s10107-023-01975-z](https://doi.org/10.1007/s10107-023-01975-z) (open access) · [arXiv:2110.06756](https://arxiv.org/abs/2110.06756)

![Convergence comparison](figures/convergence.png)

## What this is

The paper proposes and analyses **FC-GCG**, a Frank–Wolfe-type algorithm for problems of the form

$$\min_u \; F(Ku) + R(u)$$

(smooth convex loss + one-homogeneous convex regularizer) over a **Banach space** — typically a space of measures, where solutions are sparse combinations of *extremal points* of the regularizer's unit ball. The paper proves global sublinear convergence, improved to a **local linear rate** under non-degeneracy assumptions on the dual variable.

The full theory needs some heavy machinery. But the *algorithmic idea* is completely elementary, and this repo demonstrates it in the cleanest possible setting: **the lasso in ℝⁿ**,

$$\min_x \; \tfrac{1}{2}\lVert Ax - b\rVert^2 + \lambda \lVert x\rVert_1 ,$$

where the geometric picture becomes concrete:

- the regularizer's unit ball is the cross-polytope $\{\lVert x\rVert_1 \le 1\}$;
- its **extremal points are exactly the $2n$ signed coordinate vectors $\pm e_i$**;
- "sparse solution" literally means "conic combination of few extremal points".

FC-GCG then reads, in full:

1. **Insertion step** (one linear problem). Compute the dual variable $p = A^\top(b - Ax_k)$; find the extremal point most correlated with it, i.e. $i^* = \arg\max_i |p_i|$. If $\max_i |p_i| \le \lambda$, the KKT conditions hold — **stop, with a certificate of exact optimality**. Otherwise add the atom $\mathrm{sign}(p_{i^*})\, e_{i^*}$ to the active set.
2. **Fully-corrective step** (one *small* convex problem). Re-solve the lasso restricted to the active columns — a problem of dimension (#atoms) ≪ n — and drop atoms whose coefficient hits zero.

Every iterate is a combination of few extremal points: **sparsity by construction**, not as a limit.

## What the experiments show

| Script | Figure | Message |
|---|---|---|
| `experiment_convergence.py` | `figures/convergence.png` | On a 300×5000 lasso with an 11-atom solution, FC-GCG reaches **machine precision in 12 iterations** (11 insertions + 1 KKT certificate). Vanilla Frank–Wolfe decays at its classical $O(1/k)$ rate and is still at $10^{-4}$ after 2000 iterations; ISTA needs ~1000+ iterations; FISTA a few hundred. |
| `experiment_convergence.py` | `figures/sparsity.png` | Proximal-gradient iterates are **dense early on** (hundreds of nonzeros) and only sparsify near convergence; FC-GCG's support never exceeds the solution's. In genuinely infinite-dimensional problems this distinction is the whole game — dense iterates don't even exist off a grid. |
| `experiment_scaling.py` | `figures/scaling.png` | Every method pays the same $O(mn)$ per gradient/dual evaluation, so wall time is decided by **iteration counts**. FC-GCG's count is ~(#atoms), independent of $n$. Measured on this machine at $n = 200{,}000$: **FC-GCG 1.0 s**, vs FISTA 96 s, ISTA 106 s, FW 103 s — all three baselines still short of the $10^{-8}$ tolerance when their iteration caps hit, so the true gap exceeds two orders of magnitude. |

All algorithms live in `algorithms.py` (~200 lines of numpy, extensively commented): `ista`, `fista`, `frank_wolfe` (constrained formulation, step $2/(k+2)$), `fc_gcg` (with an exact coordinate-descent inner solver for the fully-corrective step).

## Run it

Requires Python ≥ 3.10, numpy, matplotlib.

```bash
python3 experiment_convergence.py     # ~20 s
python3 experiment_scaling.py         # several minutes (n up to 200,000)
```

## Honest caveats

- The lasso is a *deliberately easy* instance chosen for pedagogy: the LMO is an argmax, the fully-corrective subproblem is a tiny lasso, and well-conditioned Gaussian designs satisfy the paper's non-degeneracy assumptions comfortably. That's what makes FC-GCG's finite/linear behaviour so clean here.
- The paper's actual contribution is the analysis in **infinite dimensions** — spaces of measures where extremal points are Dirac deltas (or, in the dynamic setting of our [FoCM 2023 paper](https://doi.org/10.1007/s10208-022-09561-z), curves of measures) and grid-based methods like ISTA are unavailable or suffer grid bias. The finite-dimensional demo is a faithful shadow of the mechanism, not of the difficulty.
- Specialised lasso solvers (LARS, glmnet-style coordinate descent with screening) are also extremely fast on this instance; the comparison here is between *general-purpose first-order schemes*, which are the methods that generalise to the Banach-space setting.

## Citation

```bibtex
@article{2024-Bre-Car-Fan-Wal,
  author  = {Bredies, Kristian and Carioni, Marcello and Fanzon, Silvio
             and Walter, Daniel},
  title   = {Asymptotic linear convergence of Fully-Corrective Generalized
             Conditional Gradient methods},
  journal = {Mathematical Programming},
  volume  = {205},
  pages   = {135--202},
  year    = {2024},
  doi     = {10.1007/s10107-023-01975-z}
}
```

## License

MIT for the code in this repository. (The paper is open access, CC BY 4.0.)

## Repository files

- `algorithms.py` — the four solvers, heavily commented (start here).
- `experiment_convergence.py` — figures 1–2, runs in ~20 s.
- `experiment_scaling.py` — the scaling protocol in one script.
- `run_one.py` + `plot_scaling.py` — the same protocol split into resumable chunks (one `(n, method)` timing per invocation, cached in `results.json`) — convenient on shared machines or if a long run gets interrupted. The committed `results.json` and `figures/` were produced this way.
