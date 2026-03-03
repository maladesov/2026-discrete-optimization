from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KnapsackInstance:
    n_items: int
    capacity: int
    values: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_file(cls, path: str) -> "KnapsackInstance":
        with open(path, "r") as f:
            n, capacity = map(int, f.readline().split())
            values = np.empty(n, dtype=np.int64)
            weights = np.empty(n, dtype=np.int64)
            for i in range(n):
                v, w = map(int, f.readline().split())
                values[i] = v
                weights[i] = w
        return cls(n_items=n, capacity=capacity, values=values, weights=weights)

    def complexity(self) -> int:
        return self.n_items * self.capacity


@dataclass
class KnapsackSolution:
    selected: frozenset[int]
    objective: int

    def verify(self, instance: KnapsackInstance) -> bool:
        total_weight = sum(instance.weights[i] for i in self.selected)
        total_value = sum(instance.values[i] for i in self.selected)
        return total_weight <= instance.capacity and total_value == self.objective

    def __str__(self) -> str:
        return f"objective={self.objective}, items={len(self.selected)}"


class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, instance: KnapsackInstance) -> KnapsackSolution:
        ...
