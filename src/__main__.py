import argparse
import sys
import time

from solvers import KnapsackInstance, DPSolver

SOLVERS = {
    "dp": DPSolver,
}


def parse_args():
    parser = argparse.ArgumentParser(prog="knapsack")
    parser.add_argument("files", metavar="FILE", nargs="+")
    parser.add_argument(
        "--solver",
        choices=SOLVERS.keys(),
        default=SOLVERS.keys().__iter__().__next__(),
    )
    return parser.parse_args()


def run_file(path: str, solver):
    t0 = time.perf_counter()
    instance = KnapsackInstance.from_file(path)
    solution = solver.solve(instance)
    elapsed = time.perf_counter() - t0

    valid = solution.verify(instance)

    print(solution.objective)
    print(" ".join(str(i) for i in sorted(solution.selected)))

    status = "OK" if valid else "INVALID"
    print(f"[{path}] {solution} | {status} | {elapsed:.2f}s", file=sys.stderr)

    if not valid:
        sys.exit(1)


def main():
    args = parse_args()
    solver = SOLVERS[args.solver]()
    for path in args.files:
        run_file(path, solver)


if __name__ == "__main__":
    main()
