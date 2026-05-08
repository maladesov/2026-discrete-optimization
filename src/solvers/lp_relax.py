import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .base import (
    AbstractSolver,
    FacilityInstance,
    FacilitySolution,
    compute_objective,
)
from .greedy import GreedySolver
from .local_search import LocalSearchSolver


class LPRelaxSolver(AbstractSolver):
    def __init__(
        self,
        time_budget: float = 540.0,
        k_nearest: int = 35,
        seed: int = 0,
        ls_budget_frac: float = 0.5,
    ):
        self.time_budget = time_budget
        self.k_nearest = k_nearest
        self.seed = seed
        self.ls_budget_frac = ls_budget_frac

    def solve(self, instance: FacilityInstance) -> FacilitySolution:
        deadline = time.perf_counter() + self.time_budget
        dist = instance.distance_matrix()

        y = self._solve_lp(instance, dist, deadline)
        forced_open = self._round(instance, y)
        assignment = self._greedy_assign(instance, dist, forced_open)
        if assignment is None:
            return GreedySolver().solve(instance)

        remaining = max(deadline - time.perf_counter(), 1.0)
        ls = LocalSearchSolver(
            time_budget=remaining * self.ls_budget_frac, seed=self.seed
        )
        polished = ls.iterate_local_search(
            instance, assignment, remaining * self.ls_budget_frac, self.seed
        )
        return FacilitySolution(
            assignment=polished, objective=compute_objective(instance, polished)
        )

    def _solve_lp(self, instance, dist, deadline):
        n, m = instance.n_facilities, instance.n_customers
        K = min(self.k_nearest, n)

        nearest = np.argpartition(dist, K - 1, axis=1)[:, :K]
        pairs_c = np.repeat(np.arange(m), K)
        pairs_f = nearest.reshape(-1)
        n_pairs = pairs_c.size
        n_vars = n_pairs + n

        c_obj = np.empty(n_vars, dtype=np.float64)
        c_obj[:n_pairs] = dist[pairs_c, pairs_f]
        c_obj[n_pairs:] = instance.setup_cost

        A_eq = lil_matrix((m, n_vars), dtype=np.float64)
        for idx in range(n_pairs):
            A_eq[pairs_c[idx], idx] = 1.0
        b_eq = np.ones(m, dtype=np.float64)

        A_ub = lil_matrix((n + n_pairs, n_vars), dtype=np.float64)
        for idx in range(n_pairs):
            A_ub[pairs_f[idx], idx] = instance.demand[pairs_c[idx]]
        for f in range(n):
            A_ub[f, n_pairs + f] = -instance.capacity[f]
        for idx in range(n_pairs):
            A_ub[n + idx, idx] = 1.0
            A_ub[n + idx, n_pairs + pairs_f[idx]] = -1.0
        b_ub = np.zeros(n + n_pairs, dtype=np.float64)

        bounds = [(0.0, 1.0)] * n_vars
        time_left = max(deadline - time.perf_counter(), 5.0)

        res = linprog(
            c_obj,
            A_ub=A_ub.tocsr(),
            b_ub=b_ub,
            A_eq=A_eq.tocsr(),
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"time_limit": min(time_left, 600.0)},
        )
        if res.x is None:
            return np.full(n, 0.5)
        return np.clip(res.x[n_pairs:], 0.0, 1.0)

    @staticmethod
    def _round(instance, y):
        order = np.argsort(-y)
        total = float(instance.demand.sum())
        cum = 0.0
        for k in range(1, len(y) + 1):
            cum += instance.capacity[order[k - 1]]
            if cum >= 1.05 * total:
                return np.sort(order[:k])
        return np.sort(order)

    @staticmethod
    def _greedy_assign(instance, dist, forced_open):
        n, m = instance.n_facilities, instance.n_customers
        demand, cap = instance.demand, instance.capacity

        if cap[forced_open].sum() < demand.sum() - 1e-6:
            return None

        opened_mask = np.zeros(n, dtype=bool)
        opened_mask[forced_open] = True
        load = np.zeros(n, dtype=np.float64)
        assignment = np.full(m, -1, dtype=np.int64)

        for c in np.argsort(-demand):
            d = demand[c]
            slack = cap - load
            cost = np.where(opened_mask & (slack >= d - 1e-9), dist[c], np.inf)
            target = int(np.argmin(cost))
            if not np.isfinite(cost[target]):
                return None
            assignment[c] = target
            load[target] += d
        return assignment
