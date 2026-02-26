import argparse
import sys
import time

from solvers import SetCoverInstance, RelaxationSolver
from solvers.base import AbstractSolver


def parse_args():
    parser = argparse.ArgumentParser(
        prog="set_cover",
        description="Set Cover optimisation solver (LP relaxation + rounding)",
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="Path(s) to input file(s) in the standard Set Cover format",
    )
    parser.add_argument(
        "--solver",
        choices=["relaxation"],
        default="relaxation",
        help="Solver to use (default: relaxation)",
    )
    return parser.parse_args()


def make_solver(name: str):
    if name == "relaxation":
        return RelaxationSolver()
    raise ValueError(f"Unknown solver: {name}")


def run_file(path: str, solver: AbstractSolver):
    t0 = time.perf_counter()

    instance = SetCoverInstance.from_file(path)
    solution = solver.solve(instance)
    elapsed = time.perf_counter() - t0

    valid = solution.verify(instance)

    print(
        f"{solution.objective:.6f}",
        " ".join(str(i) for i in sorted(solution.selected)),
        sep="\n",
    )

    valid_str = "OK" if valid else "INVALID — coverage incomplete!"
    print(
        f"{solution.objective:.6f}",
        " ".join(str(i) for i in sorted(solution.selected)),
        f"[{path}] {solution} | valid={valid_str} | time={elapsed:.2f}s",

        sep="\n",
        file=sys.stderr,
    )

    if not valid:
        sys.exit(1)


def main():
    args = parse_args()
    solver = make_solver(args.solver)

    for path in args.files:
        run_file(path, solver)


if __name__ == "__main__":
    main()
