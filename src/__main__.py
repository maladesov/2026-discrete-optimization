import argparse
import concurrent.futures
import sys
import time

from solvers import (
    SimpleBBSolver,
    TSPInstance,
)

SOLVERS = {
    "simple_bb": SimpleBBSolver,
}


def parse_args():
    parser = argparse.ArgumentParser(prog="tsp")
    parser.add_argument("files", metavar="FILE", nargs="+")
    parser.add_argument(
        "--solver",
        choices=SOLVERS.keys(),
        default=next(iter(SOLVERS.keys())),
    )
    return parser.parse_args()


def run_file(path: str, solver) -> None:
    t0 = time.perf_counter()
    instance = TSPInstance.from_file(path)
    solution = solver.solve(instance)
    elapsed = time.perf_counter() - t0

    valid = solution.verify(instance)

    print(f"{solution.objective:.6f}")
    print(" ".join(str(i) for i in solution.tour))

    status = "OK" if valid else "INVALID"
    print(f"[{path}] {solution} | {status} | {elapsed:.2f}s", file=sys.stderr)

    if not valid:
        sys.exit(1)


def main() -> None:
    args = parse_args()
    solver = SOLVERS[args.solver]()

    TIMEOUT_SEC = 1800
    for path in args.files:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_file, path, solver)
            try:
                future.result(timeout=TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                print(f"Timeout: {path} exceeded {TIMEOUT_SEC} seconds", file=sys.stderr)
            except Exception as e:
                print(f"Error processing file {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
