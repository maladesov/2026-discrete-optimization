from __future__ import annotations

import numpy as np
from .base import AbstractSolver, SetCoverInstance, SetCoverSolution


class GreedySolver(AbstractSolver):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def solve(self, instance: SetCoverInstance) -> SetCoverSolution:
        n, m = instance.n_elements, instance.n_sets
        c = instance.costs.astype(np.float64)
        selected = set()
        covered = np.zeros(n, dtype=bool)

        while not covered.all():
            uncovered = set(int(j) for j in np.where(~covered)[0])
            best_i, best_ratio = -1, np.inf

            for i in range(m):
                if i in selected:
                    continue

                new_covered = len(instance.sets[i] & uncovered)
                if new_covered == 0:
                    continue

                ratio = c[i] / new_covered
                if ratio < best_ratio:

                    best_ratio, best_i = ratio, i
            if best_i == -1:
                break

            selected.add(best_i)

            for j in instance.sets[best_i]:
                covered[j] = True

        obj = float(np.sum(c[list(selected)]))

        return SetCoverSolution(
            selected=frozenset(selected),
            objective=obj,
            lp_lower_bound=obj,
            is_optimal=False,
        )
