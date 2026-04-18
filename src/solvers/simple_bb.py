import numpy as np

from .base import AbstractSolver, TSPInstance, TSPSolution


class SimpleBBSolver(AbstractSolver):
    def __init__(self, max_nodes: int = 5_000_000, matrix_size_limit: int = 5000):
        self.max_nodes = max_nodes
        self.matrix_size_limit = matrix_size_limit

    def solve(self, instance: TSPInstance) -> TSPSolution:
        n = instance.n
        coords = instance.coords

        nn_tour, nn_cost = self._nearest_neighbour(coords, n)
        best_tour: list[int] = nn_tour
        best_cost: float = nn_cost

        if n > self.matrix_size_limit:
            return TSPSolution(tour=tuple(best_tour), objective=best_cost)

        D = self._dist_matrix(coords)
        best_tour, best_cost = self._two_opt_pass(D, best_tour, best_cost)

        positive = D[D > 0]
        min_edge = float(positive.min()) if positive.size else 0.0

        path: list[int] = [0]
        visited = np.zeros(n, dtype=bool)
        visited[0] = True

        nodes = 0

        def dfs(cur: int, cost: float) -> None:
            nonlocal nodes, best_cost, best_tour
            nodes += 1
            if nodes > self.max_nodes:
                return
            k = len(path)
            if k == n:
                total = cost + float(D[cur, 0])
                if total < best_cost:
                    best_cost = total
                    best_tour = list(path)
                return
            remaining = n - k + 1
            if cost + remaining * min_edge >= best_cost:
                return

            row = D[cur]
            order = np.argsort(row)
            for j_arr in order:
                j = int(j_arr)
                if visited[j] or j == cur:
                    continue
                dij = float(row[j])
                if cost + dij + (n - k) * min_edge >= best_cost:
                    break
                visited[j] = True
                path.append(j)
                dfs(j, cost + dij)
                path.pop()
                visited[j] = False
                if nodes > self.max_nodes:
                    return

        dfs(0, 0.0)

        final_cost = self._tour_cost(coords, best_tour)
        return TSPSolution(tour=tuple(best_tour), objective=final_cost)

    @staticmethod
    def _tour_cost(coords: np.ndarray, tour: list[int]) -> float:
        idx = np.asarray(tour, dtype=np.int64)
        nxt = np.roll(idx, -1)
        dx = coords[idx, 0] - coords[nxt, 0]
        dy = coords[idx, 1] - coords[nxt, 1]
        return float(np.sum(np.hypot(dx, dy)))

    @staticmethod
    def _dist_matrix(coords: np.ndarray) -> np.ndarray:
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    @staticmethod
    def _nearest_neighbour(coords: np.ndarray, n: int, start: int = 0) -> tuple[list[int], float]:
        visited = np.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True
        cost = 0.0
        cur = start
        for _ in range(n - 1):
            dx = coords[:, 0] - coords[cur, 0]
            dy = coords[:, 1] - coords[cur, 1]
            dists = np.hypot(dx, dy)
            dists[visited] = np.inf
            nxt = int(np.argmin(dists))
            tour.append(nxt)
            cost += float(dists[nxt])
            visited[nxt] = True
            cur = nxt
        cost += float(np.hypot(coords[tour[0], 0] - coords[cur, 0], coords[tour[0], 1] - coords[cur, 1]))
        return tour, cost

    @staticmethod
    def _two_opt_pass(D: np.ndarray, tour: list[int], cost: float) -> tuple[list[int], float]:
        n = len(tour)
        improved = True
        t = tour[:]
        while improved:
            improved = False
            for i in range(n - 1):
                a, b = t[i], t[i + 1]
                for j in range(i + 2, n):
                    c = t[j]
                    d = t[(j + 1) % n]
                    if a == d:
                        continue
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        cost += float(delta)
                        improved = True
        return t, cost
