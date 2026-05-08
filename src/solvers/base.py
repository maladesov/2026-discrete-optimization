from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VRPInstance:
    n_points: int
    n_vehicles: int
    capacity: int
    demand: np.ndarray
    xy: np.ndarray

    @classmethod
    def from_file(cls, path: str) -> "VRPInstance":
        with open(path, "r") as f:
            tokens = f.read().split()
        it = iter(tokens)
        n = int(next(it))
        v = int(next(it))
        c = int(next(it))
        demand = np.empty(n, dtype=np.float64)
        xy = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            demand[i] = float(next(it))
            xy[i, 0] = float(next(it))
            xy[i, 1] = float(next(it))
        return cls(n_points=n, n_vehicles=v, capacity=c, demand=demand, xy=xy)

    def distance_matrix(self) -> np.ndarray:
        diff = self.xy[:, None, :] - self.xy[None, :, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


@dataclass
class VRPSolution:
    routes: list[list[int]]
    objective: float

    def verify(self, instance: VRPInstance, tol: float = 1e-6) -> bool:
        if sum(1 for r in self.routes if r) > instance.n_vehicles:
            return False
        seen: set[int] = set()
        for r in self.routes:
            load = 0.0
            for c in r:
                if c <= 0 or c >= instance.n_points or c in seen:
                    return False
                seen.add(c)
                load += float(instance.demand[c])
            if load > instance.capacity + tol:
                return False
        if seen != set(range(1, instance.n_points)):
            return False
        D = instance.distance_matrix()
        true_obj = sum(route_distance(D, r) for r in self.routes)
        return abs(true_obj - self.objective) < max(1.0, abs(self.objective)) * 1e-6

    def __str__(self) -> str:
        used = sum(1 for r in self.routes if r)
        return f"objective={self.objective:.4f}, routes={used}"


class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, instance: VRPInstance) -> VRPSolution:
        ...


def route_distance(D: np.ndarray, route: list[int]) -> float:
    if not route:
        return 0.0
    d = float(D[0, route[0]] + D[route[-1], 0])
    for k in range(len(route) - 1):
        d += float(D[route[k], route[k + 1]])
    return d


def total_distance(D: np.ndarray, routes: list[list[int]]) -> float:
    return sum(route_distance(D, r) for r in routes)


def route_load(instance: VRPInstance, route: list[int]) -> float:
    return float(sum(instance.demand[c] for c in route))
