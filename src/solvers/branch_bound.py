import numpy as np
from .base import AbstractSolver, KnapsackInstance, KnapsackSolution


class BBSolver(AbstractSolver):
    def __init__(self, max_nodes: int = 50_000_000):
        self.max_nodes = max_nodes

    def solve(self, instance: KnapsackInstance) -> KnapsackSolution:
        n = instance.n_items
        W = instance.capacity

        ratios = instance.values / np.maximum(instance.weights, 1e-9)
        order = np.argsort(-ratios)
        values = instance.values[order]
        weights = instance.weights[order]

        best_value, best_selected = self._greedy(values, weights, W, n)

        stack: list[tuple[int, int, int, tuple[int, ...]]] = [(0, 0, 0, ())]
        nodes = 0

        while stack:
            nodes += 1
            if nodes > self.max_nodes:
                break

            level, value, weight, selected = stack.pop()

            if level == n:
                if value > best_value:
                    best_value = value
                    best_selected = selected
                continue

            bound = self._bound(level, value, weight, values, weights, W, n)
            if bound <= best_value:
                continue

            stack.append((level + 1, value, weight, selected))

            if weight + weights[level] <= W:
                new_val = value + values[level]
                new_wt = weight + weights[level]
                new_sel = selected + (level,)
                if new_val > best_value:
                    best_value = new_val
                    best_selected = new_sel
                stack.append((level + 1, new_val, new_wt, new_sel))

        original = tuple(order[i] for i in best_selected)

        return KnapsackSolution(
            selected=frozenset(original),
            objective=int(best_value),
        )

    def _bound(self, level, value, weight, values, weights, W, n):
        bound = float(value)
        rem = W - weight
        for i in range(level, n):
            if weights[i] <= rem:
                bound += values[i]
                rem -= weights[i]
            else:
                bound += values[i] * (rem / weights[i])
                break
        return bound

    def _greedy(self, values, weights, W, n):
        total_val = 0
        total_wt = 0
        selected = []
        for i in range(n):
            if total_wt + weights[i] <= W:
                total_wt += weights[i]
                total_val += values[i]
                selected.append(i)
        return total_val, tuple(selected)
