import numpy as np
from .base import (
    AbstractSolver,
    FacilityInstance,
    FacilitySolution,
    compute_objective,
)


class GreedySolver(AbstractSolver):
    def solve(self, instance: FacilityInstance) -> FacilitySolution:
        n = instance.n_facilities
        m = instance.n_customers
        dist = instance.distance_matrix()

        opened = np.zeros(n, dtype=bool)
        remaining = instance.capacity.copy()
        assignment = np.full(m, -1, dtype=np.int64)

        order = np.argsort(-instance.demand)

        for c in order:
            d = instance.demand[c]
            feasible = remaining >= d
            if not feasible.any():
                raise RuntimeError("infeasible")
            cost = dist[c].copy()
            cost = cost + np.where(opened, 0.0, instance.setup_cost)
            cost = np.where(feasible, cost, np.inf)
            f = int(np.argmin(cost))
            assignment[c] = f
            remaining[f] -= d
            opened[f] = True

        objective = compute_objective(instance, assignment)
        return FacilitySolution(assignment=assignment, objective=objective)
