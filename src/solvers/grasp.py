import time
import numpy as np

from .base import (
    AbstractSolver,
    FacilityInstance,
    FacilitySolution,
    compute_objective,
)
from .greedy import GreedySolver
from .local_search import loads, local_optimum


class GRASPSolver(AbstractSolver):
    def __init__(
        self,
        time_budget: float = 540.0,
        alpha: float = 0.3,
        n_starts: int = 400,
        seed: int = 0,
    ):
        self.time_budget = time_budget
        self.alpha = alpha
        self.n_starts = n_starts
        self.seed = seed

    def solve(self, instance: FacilityInstance) -> FacilitySolution:
        deadline = time.perf_counter() + self.time_budget
        rng = np.random.default_rng(self.seed)
        dist = instance.distance_matrix()

        best_assignment = None
        best_obj = float("inf")

        for _ in range(self.n_starts):
            if time.perf_counter() >= deadline:
                break

            assignment = self._construct(instance, dist, rng)
            if assignment is None:
                continue

            load, opened = loads(instance, assignment)
            local_optimum(instance, dist, assignment, load, opened, deadline)
            obj = compute_objective(instance, assignment)
            if obj < best_obj:
                best_obj = obj
                best_assignment = assignment.copy()

        if best_assignment is None:
            return GreedySolver().solve(instance)

        return FacilitySolution(assignment=best_assignment, objective=best_obj)

    def _construct(self, instance, dist, rng):
        n = instance.n_facilities
        cap, demand, setup = instance.capacity, instance.demand, instance.setup_cost
        opened = np.zeros(n, dtype=bool)
        load = np.zeros(n, dtype=np.float64)
        assignment = np.full(instance.n_customers, -1, dtype=np.int64)

        for c in np.argsort(-demand):
            d = demand[c]
            slack = cap - load
            cost = dist[c] + np.where(opened, 0.0, setup)
            cost = np.where(slack >= d - 1e-9, cost, np.inf)

            finite_mask = np.isfinite(cost)

            if not finite_mask.any():
                return None

            cmin = cost[finite_mask].min()
            cmax = cost[finite_mask].max()
            threshold = cmin + self.alpha * (cmax - cmin)
            rcl = np.where(cost <= threshold + 1e-9)[0]
            target = int(rng.choice(rcl))

            assignment[c] = target
            load[target] += d
            opened[target] = True
        return assignment
