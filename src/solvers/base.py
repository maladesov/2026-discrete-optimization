from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FacilityInstance:
    n_facilities: int
    n_customers: int
    setup_cost: np.ndarray
    capacity: np.ndarray
    fac_xy: np.ndarray
    demand: np.ndarray
    cust_xy: np.ndarray

    @classmethod
    def from_file(cls, path: str) -> "FacilityInstance":
        with open(path, "r") as f:
            tokens = f.read().split()
        it = iter(tokens)
        n = int(next(it))
        m = int(next(it))
        setup_cost = np.empty(n, dtype=np.float64)
        capacity = np.empty(n, dtype=np.float64)
        fac_xy = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            setup_cost[i] = float(next(it))
            capacity[i] = float(next(it))
            fac_xy[i, 0] = float(next(it))
            fac_xy[i, 1] = float(next(it))
        demand = np.empty(m, dtype=np.float64)
        cust_xy = np.empty((m, 2), dtype=np.float64)
        for j in range(m):
            demand[j] = float(next(it))
            cust_xy[j, 0] = float(next(it))
            cust_xy[j, 1] = float(next(it))
        return cls(
            n_facilities=n,
            n_customers=m,
            setup_cost=setup_cost,
            capacity=capacity,
            fac_xy=fac_xy,
            demand=demand,
            cust_xy=cust_xy,
        )

    def distance_matrix(self) -> np.ndarray:
        diff = self.cust_xy[:, None, :] - self.fac_xy[None, :, :]
        return np.sqrt(np.einsum("cfd,cfd->cf", diff, diff))


@dataclass
class FacilitySolution:
    assignment: np.ndarray
    objective: float

    def verify(self, instance: FacilityInstance, tol: float = 1e-6) -> bool:
        a = self.assignment
        if a.shape != (instance.n_customers,):
            return False
        if a.min() < 0 or a.max() >= instance.n_facilities:
            return False
        load = np.zeros(instance.n_facilities, dtype=np.float64)
        np.add.at(load, a, instance.demand)
        if np.any(load > instance.capacity + tol):
            return False
        opened = np.unique(a)
        setup = float(instance.setup_cost[opened].sum())
        dx = instance.cust_xy[:, 0] - instance.fac_xy[a, 0]
        dy = instance.cust_xy[:, 1] - instance.fac_xy[a, 1]
        travel = float(np.sqrt(dx * dx + dy * dy).sum())
        return abs((setup + travel) - self.objective) < max(1.0, abs(self.objective)) * 1e-6

    def __str__(self) -> str:
        n_open = len(np.unique(self.assignment))
        return f"objective={self.objective:.4f}, open={n_open}"


class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, instance: FacilityInstance) -> FacilitySolution:
        ...


def compute_objective(
    instance: FacilityInstance, assignment: np.ndarray
) -> float:
    opened = np.unique(assignment)
    setup = float(instance.setup_cost[opened].sum())
    dx = instance.cust_xy[:, 0] - instance.fac_xy[assignment, 0]
    dy = instance.cust_xy[:, 1] - instance.fac_xy[assignment, 1]
    travel = float(np.sqrt(dx * dx + dy * dy).sum())
    return setup + travel
