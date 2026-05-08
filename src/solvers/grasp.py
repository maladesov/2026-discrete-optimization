import time
import random
import numpy as np
from .base import (
    AbstractSolver,
    VRPInstance,
    VRPSolution,
    total_distance,
    route_load,
)
from .greedy import clarke_wright, enforce_vehicle_count
from .local_search import descent


def randomized_savings(
    instance: VRPInstance, D: np.ndarray, alpha: float, rng: random.Random
) -> list:
    n = instance.n_points
    pairs = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = float(D[0, i] + D[0, j] - D[i, j])
            if s > 0:
                pairs.append((s, i, j))
    if not pairs:
        return []
    s_max = max(p[0] for p in pairs)
    s_min = min(p[0] for p in pairs)
    noise = alpha * (s_max - s_min)
    pairs = [(s + rng.uniform(-noise, noise), i, j) for s, i, j in pairs]
    pairs.sort(reverse=True)
    return pairs


class GRASPSolver(AbstractSolver):
    def __init__(
        self,
        n_starts: int = 50,
        alpha: float = 0.2,
        time_limit: float = 540.0,
        seed: int = 42,
    ):
        self.n_starts = n_starts
        self.alpha = alpha
        self.time_limit = time_limit
        self.seed = seed

    def solve(self, instance: VRPInstance) -> VRPSolution:
        rng = random.Random(self.seed)
        D = instance.distance_matrix()
        best_routes: list[list[int]] | None = None
        best_obj = float("inf")
        deadline = time.perf_counter() + self.time_limit
        for _ in range(self.n_starts):
            if time.perf_counter() >= deadline:
                break
            pairs = randomized_savings(instance, D, self.alpha, rng)
            routes = clarke_wright(instance, D, pairs=pairs)
            routes = enforce_vehicle_count(routes, instance)
            loads = [route_load(instance, r) for r in routes]
            descent(routes, loads, D, instance)
            obj = total_distance(D, routes)
            if obj < best_obj:
                best_obj = obj
                best_routes = [list(r) for r in routes]
        assert best_routes is not None
        return VRPSolution(routes=best_routes, objective=best_obj)
