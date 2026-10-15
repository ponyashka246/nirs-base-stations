"""Вероятностное покрытие (утверждение 2): p_jb = Phi((L* - L_jb)/sigma), F(S) = sum w_b [1 - (1-p0_b) prod (1-p_jb)].
Сравнение решений, оптимизированных по пороговой и вероятностной целевой функции, на обеих функциях."""
import numpy as np
import pandas as pd
from scipy.stats import norm
import heapq
from common import load, build, OPS

TH, SIGMA, K = 135.0, 8.0, 50
cells, cand, pairs = load("f1800")
W = cells["pop"].values
rows = []
for op in OPS:
    has = cand.ops.str.contains(op).values & cand.kind.eq("existing").values
    p = pairs.copy()
    p["pr"] = norm.cdf((TH - p.pl.values) / SIGMA)
    # текущая сеть: вероятность, что ячейка не обслужена ни одной станцией оператора
    po = p[has[p.c.values]]
    q0 = np.ones(len(cells))
    np.multiply.at(q0, po.b.values, 1 - po.pr.values)
    allowed = np.where(~has)[0]
    p2 = p[~has[p.c.values]].sort_values("c")
    st = np.searchsorted(p2.c.values, np.arange(len(cand) + 1))
    cc = [p2.b.values[st[j]:st[j + 1]] for j in range(len(cand))]
    cp = [p2.pr.values[st[j]:st[j + 1]] for j in range(len(cand))]

    def gain(j, q):
        c = cc[j]
        return float((W[c] * q[c] * cp[j]).sum())

    def lazy(K):
        q = q0.copy(); heap = [(-gain(j, q), j, 0) for j in allowed]; heapq.heapify(heap); S = []
        while len(S) < K:
            g, j, s = heapq.heappop(heap)
            if s == len(S):
                S.append(j); q[cc[j]] *= (1 - cp[j]); continue
            heapq.heappush(heap, (-gain(j, q), j, len(S)))
        return S

    def F(S):
        q = q0.copy()
        for j in S:
            q[cc[j]] *= (1 - cp[j])
        return float((W * (q0 - q)).sum())

    Sprob = lazy(K)
    P, al, own = build(cells, cand, pairs, op, "A", TH)
    from opt import lazy_greedy
    Sth, vals, _ = lazy_greedy(P, K=K, allowed=al)
    rows.append(dict(op=op, F_prob=F(Sprob), F_thr=F(Sth), f_thr=vals[-1], f_prob=P.value(Sprob),
                     overlap=len(set(Sprob) & set(Sth)) / K, expected_base=float((W * (1 - q0)).sum())))
    print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r["prob_gain_vs_thr"] = r.F_prob / r.F_thr - 1
r["thr_loss_on_prob"] = 1 - r.f_prob / r.f_thr
r.to_csv("out/prob.csv", index=False)
print(r.round(3).to_string())
