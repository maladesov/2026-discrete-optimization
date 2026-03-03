from .base import AbstractSolver, KnapsackInstance, KnapsackSolution


class SimpleBBSolver(AbstractSolver):
    def __init__(self, max_nodes: int = 50_000_000):
        self.max_nodes = max_nodes

    def solve(self, instance: KnapsackInstance) -> KnapsackSolution:
        n = instance.n_items
        W = instance.capacity
        values = instance.values
        weights = instance.weights

        best_value = 0
        best_selected: tuple[int, ...] = ()

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

            upper = value
            rem = W - weight
            for i in range(level, n):
                if weights[i] <= rem:
                    upper += values[i]

            if upper <= best_value:
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

        return KnapsackSolution(
            selected=frozenset(best_selected),
            objective=int(best_value),
        )
