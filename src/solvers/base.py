from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ColoringInstance:
    n_vertices: int
    n_edges: int
    adjacency: list[set[int]]  # adjacency[v] = set of neighbors of v

    @classmethod
    def from_file(cls, path: str) -> "ColoringInstance":
        with open(path, "r") as f:
            n, m = map(int, f.readline().split())
            adjacency: list[set[int]] = [set() for _ in range(n)]
            for _ in range(m):
                u, v = map(int, f.readline().split())
                adjacency[u].add(v)
                adjacency[v].add(u)
        return cls(n_vertices=n, n_edges=m, adjacency=adjacency)

    def max_degree(self) -> int:
        return max(len(adj) for adj in self.adjacency)


@dataclass
class ColoringSolution:
    colors: list[int]  # colors[v] = color of vertex v (0-indexed colors)
    n_colors: int

    def verify(self, instance: ColoringInstance) -> bool:
        if len(self.colors) != instance.n_vertices:
            return False
        if self.n_colors != max(self.colors) + 1:
            return False
        for v in range(instance.n_vertices):
            for u in instance.adjacency[v]:
                if self.colors[v] == self.colors[u]:
                    return False
        return True

    def __str__(self) -> str:
        return f"n_colors={self.n_colors}"


class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, instance: ColoringInstance) -> ColoringSolution:
        ...
