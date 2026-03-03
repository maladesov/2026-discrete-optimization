import numpy as np
from .base import AbstractSolver, KnapsackInstance, KnapsackSolution


class DPSolver(AbstractSolver):
    def solve(self, instance: KnapsackInstance) -> KnapsackSolution:
        n = instance.n_items
        W = instance.capacity
        values = instance.values
        weights = instance.weights

        dp = np.zeros((n + 1, W + 1), dtype=np.int64)

        for i in range(1, n + 1):
            w_i = weights[i - 1]
            v_i = values[i - 1]
            dp[i, :] = dp[i - 1, :]
            if w_i <= W:
                for w in range(w_i, W + 1):
                    if dp[i - 1, w - w_i] + v_i > dp[i, w]:
                        dp[i, w] = dp[i - 1, w - w_i] + v_i

        selected = []
        w = W
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i - 1, w]:
                selected.append(i - 1)
                w -= weights[i - 1]

        return KnapsackSolution(
            selected=frozenset(selected),
            objective=int(dp[n, W]),
        )
