from __future__ import annotations

import numpy as np

from .base import AbstractSolver, SetCoverInstance, SetCoverSolution


class RelaxationSolver(AbstractSolver):
    def __init__(
        self,
        verbose: bool = False,
        max_iter: int = 3000,
        tol: float = 1e-6,
        step_init: float = 2.0,
    ):
        super().__init__(verbose)
        self.max_iter = max_iter
        self.tol = tol
        self.step_init = step_init

    def solve(self, instance: SetCoverInstance) -> SetCoverSolution:
        n, m = instance.n_elements, instance.n_sets
        c = instance.costs.astype(np.float64)
        A = instance.coverage_matrix()

        lam = np.ones(n, dtype=np.float64)
        best_lb = -np.inf
        best_x = np.zeros(m)

        best_primal_obj = np.inf
        best_primal: set[int] = set()

        no_improve = 0

        for it in range(self.max_iter):
            rc = c - A.T @ lam
            x = (rc < 0.0).astype(np.float64)
            lb = float(np.sum(lam)) + float(np.sum(np.minimum(0.0, rc)))

            if lb > best_lb + self.tol:
                best_lb = lb
                best_x = x.copy()
                no_improve = 0
            else:
                no_improve += 1

            coverage = A @ x
            g = 1.0 - coverage

            g_norm_sq = float(np.dot(g, g))
            if g_norm_sq < self.tol**2:
                break

            step = self.step_init / np.sqrt(it + 1)
            lam = np.maximum(0.0, lam + step * g)

            if it % 100 == 0 or it == self.max_iter - 1:
                primal = self._primal_from_x(x, instance, c, n)
                pobj = float(np.sum(c[list(primal)])) if primal else np.inf
                if pobj < best_primal_obj:
                    best_primal_obj = pobj
                    best_primal = primal

            if best_primal_obj < np.inf:
                gap = (best_primal_obj - best_lb) / max(1.0, abs(best_lb))
                if gap < self.tol:
                    break

        freq = instance.element_frequencies()
        selected: set[int] = set()
        for i in range(m):
            s = instance.sets[i]
            if not s:
                continue
            f_i = int(max(freq[j] for j in s))
            threshold = 1.0 / max(f_i, 1)
            if best_x[i] >= threshold - self.tol:
                selected.add(i)

        selected = self._ensure_coverage(selected, instance, c, n)

        if best_primal:
            best_primal = self._ensure_coverage(best_primal, instance, c, n)
            obj_rounding = float(np.sum(c[list(selected)]))
            obj_primal = float(np.sum(c[list(best_primal)]))
            if obj_primal < obj_rounding:
                selected = best_primal

        selected = self._remove_redundant(selected, instance)
        selected = self._local_search(selected, instance, c, n)

        obj = float(np.sum(c[list(selected)]))
        is_optimal = abs(obj - best_lb) <= self.tol * max(1.0, abs(best_lb))

        return SetCoverSolution(
            selected=frozenset(selected),
            objective=obj,
            lp_lower_bound=best_lb,
            is_optimal=is_optimal,
        )

    def _local_search(
        self,
        selected: set[int],
        instance: SetCoverInstance,
        c: np.ndarray,
        n: int,
    ) -> set[int]:
        improved = True
        selected = set(selected)
        while improved:
            improved = False
            for i in list(selected):
                trial = selected - {i}
                covered = np.zeros(n, dtype=bool)
                for s in trial:
                    for j in instance.sets[s]:
                        covered[j] = True
                if covered.all():
                    cost_trial = float(np.sum(c[list(trial)]))
                    cost_selected = float(np.sum(c[list(selected)]))
                    if cost_trial <= cost_selected:
                        selected = trial
                        improved = True
                        break
        return selected

    def _primal_from_x(
        self,
        x_lag: np.ndarray,
        instance: SetCoverInstance,
        c: np.ndarray,
        n: int,
    ) -> set[int]:
        selected: set[int] = set(int(i) for i in np.where(x_lag > 0.5)[0])
        covered = np.zeros(n, dtype=bool)
        for i in selected:
            for j in instance.sets[i]:
                covered[j] = True
        return self._greedy_cover(selected, covered, instance, c)

    def _ensure_coverage(
        self,
        selected: set[int],
        instance: SetCoverInstance,
        c: np.ndarray,
        n: int,
    ) -> set[int]:
        covered = np.zeros(n, dtype=bool)
        for i in selected:
            for j in instance.sets[i]:
                covered[j] = True
        if covered.all():
            return set(selected)
        return self._greedy_cover(set(selected), covered, instance, c)

    def _greedy_cover(
        self,
        selected: set[int],
        covered: np.ndarray,
        instance: SetCoverInstance,
        c: np.ndarray,
    ) -> set[int]:
        selected = set(selected)
        while not covered.all():
            uncovered = set(int(j) for j in np.where(~covered)[0])
            best_i, best_ratio = -1, np.inf
            for i in range(instance.n_sets):
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
        return selected

    def _remove_redundant(
        self,
        selected: set[int],
        instance: SetCoverInstance,
    ) -> set[int]:
        n = instance.n_elements
        selected = set(selected)
        for i in sorted(selected, key=lambda k: -instance.costs[k]):
            remaining = selected - {i}
            covered: set[int] = set()
            for r in remaining:
                covered |= instance.sets[r]
            if len(covered) == n:
                selected.remove(i)
        return selected
