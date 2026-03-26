import heapq
from .base import AbstractSolver, ColoringInstance, ColoringSolution


class DSaturSolver(AbstractSolver):
    """DSatur: на каждом шаге выбираем неокрашенную вершину
    с максимальной насыщенностью (число различных цветов у соседей),
    при равенстве — с максимальной степенью.
    Назначаем минимальный допустимый цвет."""

    def solve(self, instance: ColoringInstance) -> ColoringSolution:
        n = instance.n_vertices
        adj = instance.adjacency

        colors = [-1] * n
        saturation = [0] * n  # число различных цветов среди соседей
        neighbor_colors: list[set[int]] = [set() for _ in range(n)]

        # heap: (-saturation, -degree, vertex)
        heap = [(-0, -len(adj[v]), v) for v in range(n)]
        heapq.heapify(heap)

        colored = 0
        while colored < n:
            # Берём вершину с максимальной насыщенностью
            while heap:
                neg_sat, neg_deg, v = heapq.heappop(heap)
                if colors[v] == -1 and -neg_sat == saturation[v]:
                    break
            else:
                break

            # Назначаем минимальный допустимый цвет
            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            colored += 1

            # Обновляем насыщенность соседей
            for u in adj[v]:
                if colors[u] == -1 and c not in neighbor_colors[u]:
                    neighbor_colors[u].add(c)
                    saturation[u] += 1
                    heapq.heappush(heap, (-saturation[u], -len(adj[u]), u))

        n_colors = max(colors) + 1
        return ColoringSolution(colors=colors, n_colors=n_colors)
