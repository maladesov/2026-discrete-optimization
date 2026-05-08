import numpy as np
from .base import (
    AbstractSolver,
    VRPInstance,
    VRPSolution,
    total_distance,
)


def two_opt(route: list[int], D: np.ndarray) -> list[int]:
    n = len(route)
    if n < 4:
        return list(route)
    r = list(route)
    while True:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a = r[i - 1] if i > 0 else 0
                b = r[i]
                c = r[j]
                d = r[j + 1] if j + 1 < n else 0
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-9:
                    r[i : j + 1] = r[i : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return r


def clarke_wright(
    instance: VRPInstance,
    D: np.ndarray,
    pairs: list | None = None,
) -> list[list[int]]:
    n, cap = instance.n_points, instance.capacity
    routes: dict[int, list[int]] = {i: [i] for i in range(1, n)}
    demand: dict[int, float] = {i: float(instance.demand[i]) for i in range(1, n)}
    where: dict[int, int] = {i: i for i in range(1, n)}

    if pairs is None:
        pairs = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = float(D[0, i] + D[0, j] - D[i, j])
                if s > 0:
                    pairs.append((s, i, j))
        pairs.sort(reverse=True)

    for _, i, j in pairs:
        ri, rj = where[i], where[j]
        if ri == rj:
            continue
        if demand[ri] + demand[rj] > cap:
            continue
        ra, rb = routes[ri], routes[rj]
        if ra[-1] == i and rb[0] == j:
            new = ra + rb
        elif ra[0] == i and rb[-1] == j:
            new = rb + ra
        elif ra[-1] == i and rb[-1] == j:
            new = ra + rb[::-1]
        elif ra[0] == i and rb[0] == j:
            new = ra[::-1] + rb
        else:
            continue
        routes[ri] = new
        demand[ri] += demand[rj]
        for c in rb:
            where[c] = ri
        del routes[rj]
        del demand[rj]

    return list(routes.values())


def _drain_smallest(routes, loads, smallest, instance) -> bool:
    cap = instance.capacity
    progress = False
    for c in list(routes[smallest]):
        dc = float(instance.demand[c])
        moved = False
        for ri in range(len(routes)):
            if ri == smallest:
                continue
            if loads[ri] + dc <= cap:
                routes[smallest].remove(c)
                routes[ri].append(c)
                loads[smallest] -= dc
                loads[ri] += dc
                moved = True
                progress = True
                break
        if moved:
            continue
        for ri in range(len(routes)):
            if ri == smallest:
                continue
            swapped = False
            for c2 in list(routes[ri]):
                d2 = float(instance.demand[c2])
                if d2 >= dc:
                    continue
                if loads[ri] - d2 + dc > cap:
                    continue
                routes[smallest].remove(c)
                routes[smallest].append(c2)
                routes[ri].remove(c2)
                routes[ri].append(c)
                loads[smallest] += d2 - dc
                loads[ri] += dc - d2
                progress = True
                swapped = True
                break
            if swapped:
                break
    return progress


def enforce_vehicle_count(
    routes: list[list[int]], instance: VRPInstance
) -> list[list[int]]:
    routes = [r for r in routes if r]
    V = instance.n_vehicles
    while len(routes) > V:
        loads = [sum(float(instance.demand[c]) for c in r) for r in routes]
        smallest = min(range(len(routes)), key=lambda i: loads[i])
        if not _drain_smallest(routes, loads, smallest, instance):
            break
        if not routes[smallest]:
            del routes[smallest]
    while len(routes) < V:
        routes.append([])
    return routes


class GreedySolver(AbstractSolver):
    def solve(self, instance: VRPInstance) -> VRPSolution:
        D = instance.distance_matrix()
        routes = clarke_wright(instance, D)
        routes = enforce_vehicle_count(routes, instance)
        routes = [two_opt(r, D) for r in routes]
        return VRPSolution(routes=routes, objective=total_distance(D, routes))
