from .base import AbstractSolver, ColoringInstance, ColoringSolution


class GreedySolver(AbstractSolver):
    """Жадная раскраска: вершины сортируются по убыванию степени,
    каждой назначается минимальный допустимый цвет."""

    def solve(self, instance: ColoringInstance) -> ColoringSolution:
        n = instance.n_vertices
        adj = instance.adjacency

        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

        colors = [-1] * n
        for v in order:
            used = {colors[u] for u in adj[v] if colors[u] != -1}
            c = 0
            while c in used:
                c += 1
            colors[v] = c

        n_colors = max(colors) + 1
        return ColoringSolution(colors=colors, n_colors=n_colors)
