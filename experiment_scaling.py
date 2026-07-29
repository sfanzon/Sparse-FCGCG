"""Run the canonical dimension-scaling protocol in one batch.

The experiment definition is shared with run_one.py through
scaling_protocol.py. This command writes results.json and scaling.png only
after every configured task has completed.

Run: python3 experiment_scaling.py
"""

import json
from pathlib import Path

from plot_scaling import plot_results
from scaling_protocol import (
    METHODS,
    PROBLEM_SIZES,
    build_reference,
    make_instance,
    run_method,
    task_definition,
    validate_results,
)


def batch_task_definition(n, method):
    """Public seam used by the protocol-equality regression test."""
    return task_definition(n, method)


def main():
    results = {}
    for n in PROBLEM_SIZES:
        print(f"\n--- n = {n} ---")
        A, b, lam = make_instance(n)
        meta = build_reference(A, b, lam)
        row = {"meta": meta}
        for method in METHODS:
            result = run_method(method, A, b, lam, meta)
            row[method] = result
            suffix = "reached tol" if result["reached"] else "NOT reached"
            print(f"  {method:12s} {result['t']:8.2f}s   {suffix}")
        results[str(n)] = row

    validate_results(results)
    output = Path("results.json")
    with output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=1)
        stream.write("\n")
    plot_results(output, Path("figures/scaling.png"))


if __name__ == "__main__":
    main()
