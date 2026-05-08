import argparse
import sys
import concurrent.futures
import time

from solvers import VRPInstance, GreedySolver, LocalSearchSolver, GRASPSolver

SOLVERS = {
    "greedy": GreedySolver,
    "ls": LocalSearchSolver,
    "grasp": GRASPSolver,
}


def parse_args():
    parser = argparse.ArgumentParser(prog="vrp")
    parser.add_argument("files", metavar="FILE", nargs="+")
    parser.add_argument(
        "--solver",
        choices=SOLVERS.keys(),
        default=next(iter(SOLVERS.keys())),
    )
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def run_file(path: str, solver):
    t0 = time.perf_counter()
    instance = VRPInstance.from_file(path)
    solution = solver.solve(instance)
    elapsed = time.perf_counter() - t0

    valid = solution.verify(instance)

    print(f"{solution.objective:.4f} 0")
    for r in solution.routes:
        print(" ".join(["0"] + [str(int(c)) for c in r] + ["0"]))

    status = "OK" if valid else "INVALID"
    print(f"[{path}] {solution} | {status} | {elapsed:.2f}s", file=sys.stderr)


def main():
    args = parse_args()
    solver = SOLVERS[args.solver]()

    for path in args.files:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_file, path, solver)
            try:
                future.result(timeout=args.timeout)
            except concurrent.futures.TimeoutError:
                print(
                    f"Timeout: {path} exceeded {args.timeout} seconds", file=sys.stderr
                )
            except Exception as e:
                print(f"Error processing file {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
