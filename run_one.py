"""Run one canonical scaling task and cache it in results.json.

Usage: python3 run_one.py <n> <method>
       method in {ref,gcg,fw,ista,fista}
"""

import json
from pathlib import Path
import sys

from scaling_protocol import (
    METHODS,
    PROTOCOL_VERSION,
    build_reference,
    make_instance,
    run_method,
    task_definition,
)


RESULTS_PATH = Path("results.json")


def resumable_task_definition(n, method):
    """Public seam used by the protocol-equality regression test."""
    return task_definition(n, method)


def load_results(path=RESULTS_PATH):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_results(results, path=RESULTS_PATH):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=1)
        stream.write("\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: python3 run_one.py <n> <method>")
    n = int(argv[0])
    method = argv[1]
    resumable_task_definition(n, method)  # validate before allocating arrays

    results = load_results()
    key = str(n)
    results.setdefault(key, {})
    A, b, lam = make_instance(n)

    if method == "ref":
        # A new reference changes both the KKT certificate and the safe
        # Lipschitz bound, so cached timings for this dimension are invalid.
        results[key] = {"meta": build_reference(A, b, lam)}
        output = results[key]["meta"]
    else:
        if method not in METHODS:
            raise ValueError(f"unsupported method: {method}")
        if "meta" not in results[key]:
            raise RuntimeError(f"run `python3 run_one.py {n} ref` first")
        if results[key]["meta"].get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                f"cached metadata predates protocol version {PROTOCOL_VERSION}; "
                f"rerun `python3 run_one.py {n} ref`"
            )
        results[key][method] = run_method(
            method, A, b, lam, results[key]["meta"]
        )
        output = results[key][method]

    save_results(results)
    print(n, method, output)


if __name__ == "__main__":
    main()
