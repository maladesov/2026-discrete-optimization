import time
import random
import numpy as np
from .base import (
    AbstractSolver,
    VRPInstance,
    VRPSolution,
    total_distance,
    route_load,
)
from .greedy import clarke_wright, two_opt, enforce_vehicle_count


def two_opt_all(routes: list[list[int]], D: np.ndarray) -> bool:
    changed = False
    for i, r in enumerate(routes):
        new_r = two_opt(r, D)
        if new_r != r:
            routes[i] = new_r
            changed = True
    return changed


def try_relocate(routes, loads, D, instance) -> bool:
    cap = instance.capacity
    R = len(routes)
    for ai in range(R):
        ra = routes[ai]
        for k in range(len(ra)):
            c = ra[k]
            dc = float(instance.demand[c])
            pa = ra[k - 1] if k > 0 else 0
            na = ra[k + 1] if k + 1 < len(ra) else 0
            gain = D[pa, c] + D[c, na] - D[pa, na]
            for bi in range(R):
                if bi == ai:
                    continue
                if loads[bi] + dc > cap:
                    continue
                rb = routes[bi]
                for p in range(len(rb) + 1):
                    pb = rb[p - 1] if p > 0 else 0
                    nb = rb[p] if p < len(rb) else 0
                    cost = D[pb, c] + D[c, nb] - D[pb, nb]
                    if cost - gain < -1e-9:
                        del ra[k]
                        rb.insert(p, c)
                        loads[ai] -= dc
                        loads[bi] += dc
                        return True
    return False


def try_exchange(routes, loads, D, instance) -> bool:
    cap = instance.capacity
    R = len(routes)
    for ai in range(R):
        ra = routes[ai]
        for ka in range(len(ra)):
            ca = ra[ka]
            da = float(instance.demand[ca])
            pa = ra[ka - 1] if ka > 0 else 0
            na = ra[ka + 1] if ka + 1 < len(ra) else 0
            for bi in range(ai + 1, R):
                rb = routes[bi]
                for kb in range(len(rb)):
                    cb = rb[kb]
                    db = float(instance.demand[cb])
                    if loads[ai] - da + db > cap:
                        continue
                    if loads[bi] - db + da > cap:
                        continue
                    pb = rb[kb - 1] if kb > 0 else 0
                    nb = rb[kb + 1] if kb + 1 < len(rb) else 0
                    old = D[pa, ca] + D[ca, na] + D[pb, cb] + D[cb, nb]
                    new = D[pa, cb] + D[cb, na] + D[pb, ca] + D[ca, nb]
                    if new - old < -1e-9:
                        ra[ka] = cb
                        rb[kb] = ca
                        loads[ai] += db - da
                        loads[bi] += da - db
                        return True
    return False


def try_two_opt_star(routes, loads, D, instance) -> bool:
    cap = instance.capacity
    R = len(routes)
    for ai in range(R):
        ra = routes[ai]
        la = len(ra)
        pre_a = [0.0]
        for c in ra:
            pre_a.append(pre_a[-1] + float(instance.demand[c]))
        for bi in range(ai + 1, R):
            rb = routes[bi]
            lb = len(rb)
            pre_b = [0.0]
            for c in rb:
                pre_b.append(pre_b[-1] + float(instance.demand[c]))
            for i in range(la + 1):
                ba = ra[i - 1] if i > 0 else 0
                aa = ra[i] if i < la else 0
                for j in range(lb + 1):
                    bb = rb[j - 1] if j > 0 else 0
                    ab = rb[j] if j < lb else 0
                    delta = D[ba, ab] + D[bb, aa] - D[ba, aa] - D[bb, ab]
                    if delta >= -1e-9:
                        continue
                    new_a_load = pre_a[i] + (pre_b[lb] - pre_b[j])
                    new_b_load = pre_b[j] + (pre_a[la] - pre_a[i])
                    if new_a_load > cap or new_b_load > cap:
                        continue
                    routes[ai] = ra[:i] + rb[j:]
                    routes[bi] = rb[:j] + ra[i:]
                    loads[ai] = new_a_load
                    loads[bi] = new_b_load
                    return True
    return False


def descent(routes, loads, D, instance) -> None:
    while True:
        if two_opt_all(routes, D):
            continue
        if try_relocate(routes, loads, D, instance):
            continue
        if try_exchange(routes, loads, D, instance):
            continue
        if try_two_opt_star(routes, loads, D, instance):
            continue
        break


def perturb(routes, loads, D, instance, rng, frac: float = 0.15) -> bool:
    customers = [c for r in routes for c in r]
    if not customers:
        return False
    snap_routes = [list(r) for r in routes]
    snap_loads = list(loads)
    k = max(1, int(frac * len(customers)))
    removed = rng.sample(customers, k)
    for c in removed:
        for r in routes:
            if c in r:
                r.remove(c)
                break
    for i, r in enumerate(routes):
        loads[i] = route_load(instance, r)
    rng.shuffle(removed)
    cap = instance.capacity
    for c in removed:
        dc = float(instance.demand[c])
        best = None
        for ri, r in enumerate(routes):
            if loads[ri] + dc > cap:
                continue
            for p in range(len(r) + 1):
                prev = r[p - 1] if p > 0 else 0
                nxt = r[p] if p < len(r) else 0
                cost = D[prev, c] + D[c, nxt] - D[prev, nxt]
                if best is None or cost < best[0]:
                    best = (cost, ri, p)
        if best is None:
            for i in range(len(routes)):
                routes[i] = snap_routes[i]
                loads[i] = snap_loads[i]
            return False
        _, ri, p = best
        routes[ri].insert(p, c)
        loads[ri] += dc
    return True


class LocalSearchSolver(AbstractSolver):
    def __init__(self, time_limit: float = 540.0, seed: int = 42):
        self.time_limit = time_limit
        self.seed = seed

    def solve(self, instance: VRPInstance) -> VRPSolution:
        rng = random.Random(self.seed)
        D = instance.distance_matrix()
        routes = enforce_vehicle_count(clarke_wright(instance, D), instance)
        loads = [route_load(instance, r) for r in routes]
        descent(routes, loads, D, instance)
        best_routes = [list(r) for r in routes]
        best_obj = total_distance(D, routes)

        deadline = time.perf_counter() + self.time_limit
        while time.perf_counter() < deadline:
            if not perturb(routes, loads, D, instance, rng):
                continue
            descent(routes, loads, D, instance)
            obj = total_distance(D, routes)
            if obj < best_obj - 1e-9:
                best_obj = obj
                best_routes = [list(r) for r in routes]
            else:
                routes = [list(r) for r in best_routes]
                loads = [route_load(instance, r) for r in routes]

        return VRPSolution(routes=best_routes, objective=best_obj)
