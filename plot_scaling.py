"""Render figures/scaling.png from validated canonical scaling results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scaling_protocol import METHODS, validate_results


NAMES = {
    "gcg": "FC-GCG (the paper)",
    "fw": "Vanilla Frank-Wolfe",
    "ista": "ISTA (prox. gradient)",
    "fista": "FISTA (accelerated)",
}
STYLE = {
    "gcg": "#C8102E",
    "fw": "#1f77b4",
    "ista": "#7f7f7f",
    "fista": "#2ca02c",
}


def plot_results(results_path="results.json", output_path="figures/scaling.png"):
    results_path = Path(results_path)
    output_path = Path(output_path)
    with results_path.open(encoding="utf-8") as stream:
        results = validate_results(json.load(stream))
    dimensions = sorted(int(key) for key in results)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    for method in METHODS:
        timings = np.array([results[str(n)][method]["t"] for n in dimensions])
        reached = np.array([
            results[str(n)][method]["reached"] for n in dimensions
        ])
        ax.loglog(
            dimensions,
            timings,
            "-o",
            color=STYLE[method],
            lw=2.2,
            ms=6,
            label=NAMES[method],
        )
        if (~reached).any():
            ax.loglog(
                np.array(dimensions)[~reached],
                timings[~reached],
                "o",
                mfc="white",
                color=STYLE[method],
                ms=10,
                mew=2,
            )
    ax.set_xlabel("problem dimension $n$   ($m = 300$ fixed, 15-sparse truth)")
    ax.set_ylabel(r"wall time to relative gap $10^{-8}$  [s]")
    ax.set_title(
        "Time to solve the lasso vs dimension\n"
        "(open markers: iteration cap hit before reaching tolerance "
        r"$\Rightarrow$ true time is even larger)"
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"{output_path} written")
    for n in dimensions:
        row = "  ".join(
            f"{method}: {results[str(n)][method]['t']:8.2f}s"
            f"{'' if results[str(n)][method]['reached'] else '*'}"
            for method in METHODS
        )
        print(f"n={n:>7}  {row}   (*: cap hit)")


if __name__ == "__main__":
    plot_results()
