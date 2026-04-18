from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    n: int
    coords: np.ndarray

    @classmethod
    def from_file(cls, path: str) -> "TSPInstance":
        with open(path, "r") as f:
            n = int(f.readline())
            coords = np.empty((n, 2), dtype=np.float64)
            for i in range(n):
                x, y = f.readline().split()
                coords[i, 0] = float(x)
                coords[i, 1] = float(y)
        return cls(n=n, coords=coords)

    def complexity(self) -> int:
        return self.n * self.n


@dataclass
class TSPSolution:
    tour: tuple[int, ...]
    objective: float

    def verify(self, instance: TSPInstance, tol: float = 1e-4) -> bool:
        n = instance.n
        if len(self.tour) != n:
            return False
        if set(self.tour) != set(range(n)):
            return False
        coords = instance.coords
        total = 0.0
        for i in range(n):
            a = self.tour[i]
            b = self.tour[(i + 1) % n]
            total += math.hypot(coords[a, 0] - coords[b, 0], coords[a, 1] - coords[b, 1])
        return abs(total - self.objective) <= tol * max(1.0, total)

    def __str__(self) -> str:
        return f"objective={self.objective:.4f}, n={len(self.tour)}"


class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, instance: TSPInstance) -> TSPSolution:
        ...
