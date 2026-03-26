import random
import time
from collections import deque
from .base import AbstractSolver, ColoringInstance, ColoringSolution
from .dsatur import DSaturSolver


class KempeSolver(AbstractSolver):
    """DSatur + итеративное уменьшение числа цветов через Kempe-цепи.

    Алгоритм:
    1. DSatur даёт стартовую k-раскраску.
    2. Пытаемся убрать цвет k-1 (максимальный):
       a) Для каждой вершины с цветом k-1 пробуем прямое перекрашивание.
       b) Если не получилось — ищем Kempe-цепь: берём пару цветов (c1, c2),
          строим связную компоненту двуцветного подграфа от соседа, меняем
          цвета вдоль цепи. Если после свопа у вершины освободился цвет — красим.
    3. Если удалось устранить цвет k-1, пробуем k-2 и т.д.
    4. Повторяем с рандомизированным DSatur для разнообразия стартовых решений.
    """

    def __init__(self, time_limit: float = 300.0):
        self.time_limit = time_limit

    def solve(self, instance: ColoringInstance) -> ColoringSolution:
        t0 = time.perf_counter()
        n = instance.n_vertices
        adj = instance.adjacency

        # Стартовое решение через DSatur
        dsatur_sol = DSaturSolver().solve(instance)
        best_colors = dsatur_sol.colors[:]
        best_k = dsatur_sol.n_colors

        # Пробуем улучшить первое решение
        improved_colors, improved_k = self._reduce_colors(
            n, adj, best_colors[:], best_k, t0,
        )
        if improved_k < best_k:
            best_colors = improved_colors
            best_k = improved_k

        # Рестарты с рандомизированным DSatur
        no_improve_count = 0
        max_no_improve = max(200, n * 5)

        while time.perf_counter() - t0 < self.time_limit * 0.95:
            colors = self._randomized_dsatur(n, adj)
            k = max(colors) + 1

            if k < best_k:
                best_colors = colors[:]
                best_k = k
                no_improve_count = 0

            colors, k = self._reduce_colors(n, adj, colors, k, t0)
            if k < best_k:
                best_colors = colors[:]
                best_k = k
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= max_no_improve:
                break

        return ColoringSolution(colors=best_colors, n_colors=best_k)

    def _randomized_dsatur(self, n: int, adj: list[set[int]]) -> list[int]:
        """DSatur с рандомизированным тай-брейком."""
        colors = [-1] * n
        saturation = [0] * n
        neighbor_colors: list[set[int]] = [set() for _ in range(n)]

        for _ in range(n):
            best_sat = -1
            candidates = []
            for v in range(n):
                if colors[v] != -1:
                    continue
                s = saturation[v]
                if s > best_sat:
                    best_sat = s
                    candidates = [v]
                elif s == best_sat:
                    candidates.append(v)

            if len(candidates) > 1:
                weights = [len(adj[v]) + 1 for v in candidates]
                v = random.choices(candidates, weights=weights, k=1)[0]
            else:
                v = candidates[0]

            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            colors[v] = c

            for u in adj[v]:
                if colors[u] == -1 and c not in neighbor_colors[u]:
                    neighbor_colors[u].add(c)
                    saturation[u] += 1

        return colors

    def _reduce_colors(
        self, n: int, adj: list[set[int]], colors: list[int], k: int,
        t0: float,
    ) -> tuple[list[int], int]:
        """Пытаемся последовательно уменьшить число цветов."""
        while k > 1 and time.perf_counter() - t0 < self.time_limit * 0.95:
            target = k - 1
            success = self._eliminate_color(n, adj, colors, target, t0)
            if success:
                k -= 1
            else:
                break
        return colors, k

    def _eliminate_color(
        self, n: int, adj: list[set[int]], colors: list[int],
        target: int, t0: float,
    ) -> bool:
        """Пытаемся убрать цвет target, перекрасив все вершины с этим цветом."""
        vertices_with_target = [v for v in range(n) if colors[v] == target]
        random.shuffle(vertices_with_target)

        for v in vertices_with_target:
            if time.perf_counter() - t0 > self.time_limit * 0.95:
                return False

            # 1. Прямое перекрашивание
            used = {colors[u] for u in adj[v]}
            recolored = False
            for c in range(target):
                if c not in used:
                    colors[v] = c
                    recolored = True
                    break

            if not recolored:
                # 2. Kempe-chain swap
                recolored = self._try_kempe_recolor(n, adj, colors, v, target)
                if not recolored:
                    return False

        return True

    def _try_kempe_recolor(
        self, n: int, adj: list[set[int]], colors: list[int],
        v: int, target: int,
    ) -> bool:
        """Пробуем перекрасить v (цвет target) через Kempe-цепи."""
        neighbor_colors_of_v = {colors[u] for u in adj[v]}

        # Перебираем пары (c_want, c_swap): хотим дать v цвет c_want,
        # для этого свопаем цепь (c_want, c_swap) у блокирующего соседа
        candidates = list(range(target))
        random.shuffle(candidates)

        for c_want in candidates:
            if c_want not in neighbor_colors_of_v:
                colors[v] = c_want
                return True

            # c_want занят. Соседи, блокирующие c_want:
            blockers = [u for u in adj[v] if colors[u] == c_want]

            for c_swap in candidates:
                if c_swap == c_want or c_swap not in neighbor_colors_of_v:
                    continue

                # Пробуем Kempe-swap (c_want, c_swap) от каждого блокера
                for u in blockers:
                    chain = self._build_kempe_chain(adj, colors, u, c_want, c_swap)

                    # swap безопасен если v не в цепи
                    if v in chain:
                        continue

                    self._swap_chain(colors, chain, c_want, c_swap)

                    if c_want not in {colors[w] for w in adj[v]}:
                        colors[v] = c_want
                        return True

                    # Откат
                    self._swap_chain(colors, chain, c_want, c_swap)

        return False

    def _build_kempe_chain(
        self, adj: list[set[int]], colors: list[int],
        start: int, c1: int, c2: int,
    ) -> set[int]:
        """BFS: связная компонента подграфа из вершин с цветами c1/c2."""
        chain = {start}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w not in chain and colors[w] in (c1, c2):
                    chain.add(w)
                    queue.append(w)
        return chain

    def _swap_chain(
        self, colors: list[int], chain: set[int], c1: int, c2: int,
    ):
        """Меняем c1↔c2 для всех вершин в цепи."""
        for v in chain:
            if colors[v] == c1:
                colors[v] = c2
            elif colors[v] == c2:
                colors[v] = c1
