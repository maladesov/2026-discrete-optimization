from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SetCoverInstance:
    n_elements: int
    n_sets: int
    costs: np.ndarray
    sets: tuple[frozenset[int], ...]

    @classmethod
    def from_file(cls, path: str) -> "SetCoverInstance":
        with open(path, "r") as f:
            n, m = map(int, f.readline().split())
            costs = np.empty(m, dtype=np.float64)
            sets: list[frozenset[int]] = []
            for i in range(m):
                parts = list(map(int, f.readline().split()))
                costs[i] = float(parts[0])
                sets.append(frozenset(parts[1:]))
        return cls(
            n_elements=n,
            n_sets=m,
            costs=costs,
            sets=tuple(sets),
        )

    def coverage_matrix(self) -> np.ndarray:
        A = np.zeros((self.n_elements, self.n_sets), dtype=np.float64)
        for i, s in enumerate(self.sets):
            for j in s:
                A[j, i] = 1.0
        return A

    def element_frequencies(self) -> np.ndarray:
        freq = np.zeros(self.n_elements, dtype=np.int32)
        for s in self.sets:
            for j in s:
                freq[j] += 1
        return freq


@dataclass
class SetCoverSolution:
    selected: frozenset[int]
    objective: float
    lp_lower_bound: float
    is_optimal: bool

    def verify(self, instance: SetCoverInstance) -> bool:
        covered: set[int] = set()
        for i in self.selected:
            covered |= instance.sets[i]
        return covered >= set(range(instance.n_elements))

    def __str__(self) -> str:
        status = "OPTIMAL" if self.is_optimal else "FEASIBLE"
        gap_str = ""
        if self.lp_lower_bound > 0:
            gap = (self.objective - self.lp_lower_bound) / max(
                self.lp_lower_bound, 1e-9
            )
            gap_str = f", LP gap: {gap:.2%}"
        return (
            f"[{status}] objective={self.objective:.4f}, "
            f"sets_chosen={len(self.selected)}"
            f"{gap_str}"
        )


class AbstractSolver(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def solve(self, instance: SetCoverInstance) -> SetCoverSolution: ...

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
