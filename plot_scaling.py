"""plot_scaling.py -- render figures/scaling.png from results.json
(produced by run_one.py, or by experiment_scaling.py's protocol)."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("results.json"))
NS = sorted(int(k) for k in res)
NAMES = {"gcg": "FC-GCG (the paper)", "fw": "Vanilla Frank-Wolfe",
         "ista": "ISTA (prox. gradient)", "fista": "FISTA (accelerated)"}
STYLE = {"gcg": "#C8102E", "fw": "#1f77b4", "ista": "#7f7f7f",
         "fista": "#2ca02c"}

fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
for key, label in NAMES.items():
    t = np.array([res[str(n)][key]["t"] for n in NS])
    ok = np.array([res[str(n)][key]["reached"] for n in NS])
    ax.loglog(NS, t, "-o", color=STYLE[key], lw=2.2, ms=6, label=label)
    if (~ok).any():
        ax.loglog(np.array(NS)[~ok], t[~ok], "o", mfc="white",
                  color=STYLE[key], ms=10, mew=2)
ax.set_xlabel("problem dimension $n$   ($m = 300$ fixed, 15-sparse truth)")
ax.set_ylabel(r"wall time to relative gap $10^{-8}$  [s]")
ax.set_title("Time to solve the lasso vs dimension\n"
             "(open markers: iteration cap hit before reaching tolerance "
             r"$\Rightarrow$ true time is even larger)")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("figures/scaling.png")
print("figures/scaling.png written")
for n in NS:
    row = "  ".join(f"{k}: {res[str(n)][k]['t']:8.2f}s"
                    f"{'' if res[str(n)][k]['reached'] else '*'}"
                    for k in NAMES)
    print(f"n={n:>7}  {row}   (*: cap hit)")
