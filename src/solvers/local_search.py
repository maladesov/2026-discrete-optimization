import time
import numpy as np

from .base import (
    AbstractSolver,
    FacilityInstance,
    FacilitySolution,
    compute_objective,
)
from .greedy import GreedySolver


class LocalSearchSolver(AbstractSolver):
    def __init__(self, time_budget: float = 540.0, seed: int = 0):
        self.time_budget = time_budget
        self.seed = seed

    def solve(self, instance: FacilityInstance) -> FacilitySolution:
        init = GreedySolver().solve(instance).assignment
        best = self.iterate_local_search(instance, init, self.time_budget, self.seed)
        return FacilitySolution(
            assignment=best, objective=compute_objective(instance, best)
        )

    def iterate_local_search(
        self,
        instance: FacilityInstance,
        initial: np.ndarray,
        time_budget: float,
        seed: int = 0,
        perturb_frac: float = 0.05,
    ) -> np.ndarray:
        deadline = time.perf_counter() + time_budget
        rng = np.random.default_rng(seed)
        dist = instance.distance_matrix()

        assignment = initial.copy()
        load, opened = loads(instance, assignment)
        local_optimum(instance, dist, assignment, load, opened, deadline)

        best = assignment.copy()
        best_obj = compute_objective(instance, best)

        while time.perf_counter() < deadline:
            cand = best.copy()
            cand_load, cand_opened = loads(instance, cand)
            _perturb(instance, dist, cand, cand_load, cand_opened, rng, perturb_frac)
            local_optimum(instance, dist, cand, cand_load, cand_opened, deadline)
            obj = compute_objective(instance, cand)
            if obj < best_obj - 1e-6:
                best = cand.copy()
                best_obj = obj
        return best


def loads(instance, assignment):
    load = np.zeros(instance.n_facilities, dtype=np.float64)
    np.add.at(load, assignment, instance.demand)
    return load, load > 0


def local_optimum(instance, dist, assignment, load, opened, deadline):
    while time.perf_counter() < deadline:
        moved = _move_pass(instance, dist, assignment, load, opened, deadline)
        if time.perf_counter() >= deadline:
            return
        closed = _close_pass(instance, dist, assignment, load, opened, deadline)
        if not (moved or closed):
            return


def _move_pass(instance, dist, assignment, load, opened, deadline) -> bool:
    n, m = instance.n_facilities, instance.n_customers
    demand, cap, setup = instance.demand, instance.capacity, instance.setup_cost

    any_improved = False
    progress = True
    while progress and time.perf_counter() < deadline:
        progress = False
        count = np.bincount(assignment, minlength=n)

        for c in range(m):
            old = int(assignment[c])
            d = demand[c]

            travel_delta = dist[c] - dist[c, old]
            setup_delta = np.where(opened, 0.0, setup)
            if count[old] == 1:
                setup_delta -= setup[old]
            setup_delta[old] = 0.0

            new_load = load + d
            new_load[old] = load[old]
            feasible = new_load <= cap + 1e-9
            delta = np.where(feasible, travel_delta + setup_delta, np.inf)

            f = int(np.argmin(delta))
            if delta[f] < -1e-9 and f != old:
                assignment[c] = f
                load[old] -= d
                load[f] += d
                count[old] -= 1
                count[f] += 1
                if count[old] == 0:
                    opened[old] = False
                opened[f] = True
                progress = True
                any_improved = True
    return any_improved


def _close_pass(instance, dist, assignment, load, opened, deadline) -> bool:
    demand, cap, setup = instance.demand, instance.capacity, instance.setup_cost

    any_improved = False
    open_idx = np.where(opened)[0]
    for f in open_idx[np.argsort(load[open_idx])]:
        if time.perf_counter() >= deadline:
            break
        if not opened[f]:
            continue
        customers = np.where(assignment == f)[0]
        if customers.size == 0:
            continue

        new_assign = np.full(instance.n_customers, -1, dtype=np.int64)
        cand_load = load.copy()
        cand_open = opened.copy()
        cand_load[f] = 0.0
        cand_open[f] = False
        new_travel = 0.0
        feasible = True
        for c in customers[np.argsort(-demand[customers])]:
            d = demand[c]
            slack = cap - cand_load
            cost = dist[c] + np.where(cand_open, 0.0, setup)
            cost = np.where(slack >= d - 1e-9, cost, np.inf)
            cost[f] = np.inf
            target = int(np.argmin(cost))
            if not np.isfinite(cost[target]):
                feasible = False
                break
            new_assign[c] = target
            cand_load[target] += d
            cand_open[target] = True
            new_travel += dist[c, target]
        if not feasible:
            continue

        old_travel = float(dist[customers, f].sum())
        opened_now = np.where(opened)[0]
        opened_after = np.where(cand_open)[0]
        gain = (
            setup[f]
            + old_travel
            - new_travel
            - setup[np.setdiff1d(opened_after, opened_now, assume_unique=True)].sum()
        )
        if gain > 1e-6:
            for c in customers:
                target = new_assign[c]
                assignment[c] = target
                load[target] += demand[c]
            load[f] = 0.0
            opened[f] = False
            for target in np.setdiff1d(opened_after, opened_now, assume_unique=True):
                opened[target] = True
            any_improved = True
    return any_improved


def _perturb(instance, dist, assignment, load, opened, rng, perturb_frac):
    demand, cap, setup = instance.demand, instance.capacity, instance.setup_cost
    open_idx = np.where(opened)[0]
    if open_idx.size <= 1:
        return
    k = max(2, int(round(perturb_frac * open_idx.size)))
    k = min(k, open_idx.size - 1)
    to_close = rng.choice(open_idx, size=k, replace=False)

    orphans = []
    for f in to_close:
        for c in np.where(assignment == f)[0]:
            load[f] -= demand[c]
            assignment[c] = -1
            orphans.append(c)
        opened[f] = False

    rng.shuffle(orphans)
    for c in orphans:
        d = demand[c]
        slack = cap - load
        cost = dist[c] + np.where(opened, 0.0, setup)
        cost = np.where(slack >= d - 1e-9, cost, np.inf)
        for f in to_close:
            cost[f] = np.inf
        target = int(np.argmin(cost))
        if not np.isfinite(cost[target]):
            slack = cap - load
            cost = dist[c] + np.where(opened, 0.0, setup)
            cost = np.where(slack >= d - 1e-9, cost, np.inf)
            target = int(np.argmin(cost))
        assignment[c] = target
        load[target] += d
        opened[target] = True
