import subprocess
import sys
import tempfile
import os


def convert_to_dzn(input_path: str) -> str:
    with open(input_path) as f:
        n, m = map(int, f.readline().split())
        edges = []
        for _ in range(m):
            u, v = map(int, f.readline().split())
            edges.append((u + 1, v + 1))  # MiniZinc 1-indexed

    degree = [0] * (n + 1)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    max_colors = max(degree) + 1

    lines = [f"n = {n};", f"m = {m};", f"max_colors = {max_colors};", "edges = [|"]
    for u, v in edges:
        lines.append(f"  {u}, {v} |")
    lines.append("|];")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .mzn model")
    parser.add_argument("data", help="Path to input data file")
    parser.add_argument("--solver", default="chuffed")
    parser.add_argument("--time-limit", type=int, default=60000, help="ms")
    args = parser.parse_args()

    dzn_content = convert_to_dzn(args.data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dzn", delete=False) as f:
        f.write(dzn_content)
        dzn_path = f.name

    try:
        result = subprocess.run(
            [
                "minizinc",
                "--solver",
                args.solver,
                "--time-limit",
                str(args.time_limit),
                "-s",
                args.model,
                dzn_path,
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    finally:
        os.unlink(dzn_path)


if __name__ == "__main__":
    main()
