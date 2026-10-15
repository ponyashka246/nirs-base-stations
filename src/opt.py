"""Оптимизация: монотонная субмодулярная функция вида f(S) = sum_b w_b * max(v0_b, max_{j in S} v_jb) - const.
Частные случаи: охват (v in {0,1}), выигранный охват / maximum capture (v in {0, 1/2, 1})."""
import heapq
import numpy as np
import pandas as pd


class Problem:
    def __init__(self, w, v0, cand_cells, cand_vals, cost=None):
        self.w = np.asarray(w, dtype=np.float64)
        self.v0 = np.asarray(v0, dtype=np.float64)
        self.cc = cand_cells          # list of int arrays
        self.cv = cand_vals           # list of float arrays
        self.n = len(cand_cells)
        self.cost = np.ones(self.n) if cost is None else np.asarray(cost, dtype=np.float64)

    def gain(self, j, cur):
        c = self.cc[j]
        if len(c) == 0:
            return 0.0
        d = self.cv[j] - cur[c]
        return float((self.w[c] * np.clip(d, 0, None)).sum())

    def add(self, j, cur):
        c = self.cc[j]
        cur[c] = np.maximum(cur[c], self.cv[j])

    def value(self, S):
        cur = self.v0.copy()
        for j in S:
            self.add(j, cur)
        return float((self.w * (cur - self.v0)).sum())

    def standalone(self):
        return np.array([self.gain(j, self.v0) for j in range(self.n)])


def lazy_greedy(P, K=None, budget=None, ratio=False, allowed=None):
    """CELF: ленивый жадный. ratio=True — по приросту на единицу стоимости."""
    cur = P.v0.copy()
    cand = range(P.n) if allowed is None else allowed
    heap = []
    for j in cand:
        g = P.gain(j, cur)
        if g > 0:
            key = g / P.cost[j] if ratio else g
            heap.append((-key, j, 0))
    heapq.heapify(heap)
    S, spent, it, vals = [], 0.0, 0, []
    total = 0.0
    while heap:
        if K is not None and len(S) >= K:
            break
        negk, j, stamp = heapq.heappop(heap)
        if budget is not None and spent + P.cost[j] > budget + 1e-9:
            continue
        if stamp == len(S):
            g = P.gain(j, cur)
            S.append(j); P.add(j, cur); spent += P.cost[j]; total += g; vals.append(total)
            continue
        g = P.gain(j, cur)
        if g > 0:
            key = g / P.cost[j] if ratio else g
            heapq.heappush(heap, (-key, j, len(S)))
    return S, vals, cur


def online_bound(P, S, K, allowed=None):
    """Leskovec et al. 2007, Thm 4 (кардинальный случай): OPT <= f(S) + сумма K наибольших приростов к S."""
    cur = P.v0.copy()
    for j in S:
        P.add(j, cur)
    fS = float((P.w * (cur - P.v0)).sum())
    cand = range(P.n) if allowed is None else allowed
    g = np.array([P.gain(j, cur) for j in cand if j not in set(S)])
    top = np.sort(g)[::-1][:K].sum()
    return fS, fS + top


def celf_budget(P, budget, allowed=None):
    """CELF (Leskovec et al.): лучшее из жадного по приросту и жадного по приросту на рубль; гарантия 1/2(1-1/e)."""
    S1, _, c1 = lazy_greedy(P, budget=budget, ratio=False, allowed=allowed)
    S2, _, c2 = lazy_greedy(P, budget=budget, ratio=True, allowed=allowed)
    f1, f2 = P.value(S1), P.value(S2)
    return (S1, f1) if f1 >= f2 else (S2, f2)


def swap_search(P, S, allowed, max_rounds=3, sample=400, rng=None):
    """Локальный поиск одиночными обменами от жадного решения (не ухудшает его)."""
    rng = rng or np.random.default_rng(0)
    S = list(S)
    best = P.value(S)
    allowed = np.asarray(list(allowed))
    for _ in range(max_rounds):
        improved = False
        for pos in range(len(S)):
            rest = S[:pos] + S[pos + 1:]
            cur = P.v0.copy()
            for j in rest:
                P.add(j, cur)
            base = float((P.w * (cur - P.v0)).sum())
            pool = rng.choice(allowed, size=min(sample, len(allowed)), replace=False)
            gains = [(P.gain(j, cur), j) for j in pool if j not in rest]
            g, j = max(gains)
            if base + g > best + 1e-6:
                S = rest + [j]; best = base + g; improved = True
        if not improved:
            break
    return S, best


def solve_ilp(P, K, allowed, time_limit=600):
    """Точное решение MCLP/maximum capture через HiGHS. Значения v из {0, 1/2, 1}."""
    import highspy
    allowed = list(allowed)
    levels = sorted({float(x) for j in allowed for x in P.cv[j]} - {0.0})
    cells = np.unique(np.concatenate([P.cc[j] for j in allowed]))
    pos = {b: i for i, b in enumerate(cells)}
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("time_limit", float(time_limit))
    nx = len(allowed)
    lp_cols = []
    inf = highspy.kHighsInf
    # x_j
    for _ in allowed:
        h.addVar(0, 1)
    # y_{b,level}: прирост сверх v0 по ступеням
    yidx = {}
    prev = 0.0
    for lv in levels:
        step = lv - prev
        for b in cells:
            if P.v0[b] >= lv:
                continue
            yidx[(b, lv)] = h.getNumCol()
            h.addVar(0, 1)
            h.changeColCost(yidx[(b, lv)], -P.w[b] * step)
        prev = lv
    for k in range(nx):
        h.changeColIntegrality(k, highspy.HighsVarType.kInteger)
    # y_{b,lv} <= sum_{j: v_jb >= lv} x_j
    cover = {}
    for k, j in enumerate(allowed):
        for b, v in zip(P.cc[j], P.cv[j]):
            for lv in levels:
                if v >= lv and (b, lv) in yidx:
                    cover.setdefault((b, lv), []).append(k)
    for key, col in yidx.items():
        ks = cover.get(key, [])
        idx = np.array([col] + ks, dtype=np.int32)
        val = np.array([1.0] + [-1.0] * len(ks))
        h.addRow(-inf, 0.0, len(idx), idx, val)
    h.addRow(-inf, float(K), nx, np.arange(nx, dtype=np.int32), np.ones(nx))
    h.run()
    sol = h.getSolution().col_value
    info = h.getInfo()
    S = [allowed[k] for k in range(nx) if sol[k] > 0.5]
    return S, -info.objective_function_value, info.mip_dual_bound, h.getModelStatus()
