"""run_one.py -- run one (n, method) timing and append to results.json.
Usage: python3 run_one.py <n> <method>   method in {ref,gcg,fw,ista,fista}
The 'ref' step computes and caches J*, tau, L for dimension n."""

import json, os, sys, time
import numpy as np
from algorithms import ista, fista, frank_wolfe, fc_gcg, objective

M, K_TRUE, REL_TOL = 300, 15, 1e-8
CAPS = {2000: 2000, 20000: 2000, 200000: 1200}

n = int(sys.argv[1]); method = sys.argv[2]
cap = CAPS[n]

def make_instance(n):
    rng = np.random.default_rng(n)          # seed by n: same instance across calls
    A = rng.standard_normal((M, n)) / np.sqrt(M)
    x_true = np.zeros(n)
    x_true[rng.choice(n, K_TRUE, replace=False)] = rng.standard_normal(K_TRUE) * 3
    b = A @ x_true + 0.02 * rng.standard_normal(M)
    lam = 0.1 * np.abs(A.T @ b).max()
    return A, b, lam, rng

res = json.load(open("results.json")) if os.path.exists("results.json") else {}
key = str(n)
res.setdefault(key, {})

A, b, lam, rng = make_instance(n)

if method == "ref":
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    for _ in range(50):
        v = A.T @ (A @ v); v /= np.linalg.norm(v)
    L = float(v @ (A.T @ (A @ v)))
    x_star, _ = fc_gcg(A, b, lam, n_iter=500, kkt_tol=1e-12)
    res[key]["meta"] = {"J_star": objective(A, b, x_star, lam),
                        "tau": float(np.abs(x_star).sum()), "L": L,
                        "lam": float(lam)}
else:
    meta = res[key]["meta"]
    J_target = meta["J_star"] + REL_TOL * max(1.0, abs(meta["J_star"]))
    t0 = time.perf_counter()
    if method == "gcg":
        _, tr = fc_gcg(A, b, lam, n_iter=500)
    elif method == "fw":
        _, tr = frank_wolfe(A, b, meta["tau"], n_iter=cap, trace_every=10,
                            lam_for_obj=lam)
    elif method == "ista":
        _, tr = ista(A, b, lam, n_iter=cap, trace_every=10, L=meta["L"])
    elif method == "fista":
        _, tr = fista(A, b, lam, n_iter=cap, trace_every=10, L=meta["L"])
    obj = np.array(tr.obj)
    hit = np.nonzero(obj <= J_target)[0]
    if len(hit):
        res[key][method] = {"t": tr.t[int(hit[0])], "reached": True}
    else:
        res[key][method] = {"t": time.perf_counter() - t0, "reached": False}

json.dump(res, open("results.json", "w"), indent=1)
print(n, method, res[key].get(method, res[key].get("meta")))
