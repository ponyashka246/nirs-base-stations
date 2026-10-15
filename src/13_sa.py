"""Метаэвристика для сравнения: имитация отжига (обмены), вариант A, K = 50. Плюс время работы методов."""
import time
import numpy as np
import pandas as pd
from common import load, build, OPS
from opt import lazy_greedy

K, ITERS = 50, 60000
cells, cand, pairs = load("f1800")
rows = []
for op in OPS:
    P, allowed, own = build(cells, cand, pairs, op, "A", 135.0)
    al = np.array([j for j in allowed if len(P.cc[j])])
    free = P.v0 == 0
    W = P.w
    t0 = time.time(); S, vals, _ = lazy_greedy(P, K=K, allowed=al); t_lazy = time.time() - t0
    # обычный (не ленивый) жадный — для сравнения времени
    t0 = time.time()
    cur = P.v0.copy(); Sp = []
    for _ in range(K):
        g = np.array([P.gain(j, cur) for j in al]); j = al[int(np.argmax(g))]; Sp.append(j); P.add(j, cur)
    t_plain = time.time() - t0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        t0 = time.time()
        sel = list(rng.choice(al, K, replace=False))
        cnt = np.zeros(len(W), np.int32)
        for j in sel:
            cnt[P.cc[j]] += 1
        val = float(W[(cnt > 0) & free].sum())
        best = val
        T0 = val / K * 0.5
        for it in range(ITERS):
            T = T0 * (1 - it / ITERS) + 1e-6
            pos = rng.integers(K); out = sel[pos]; inn = al[rng.integers(len(al))]
            if inn in sel:
                continue
            co, ci = P.cc[out], P.cc[inn]
            cnt[co] -= 1
            loss = float(W[co][(cnt[co] == 0) & free[co]].sum())
            gain = float(W[ci][(cnt[ci] == 0) & free[ci]].sum())
            d = gain - loss
            if d >= 0 or rng.random() < np.exp(d / T):
                cnt[ci] += 1; sel[pos] = inn; val += d
                best = max(best, val)
            else:
                cnt[co] += 1
        t_sa = time.time() - t0
        rows.append(dict(op=op, seed=seed, greedy=vals[-1], sa_best=best, sa_final=val, ratio=best / vals[-1],
                         t_lazy=t_lazy, t_plain=t_plain, t_sa=t_sa))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r.to_csv("out/sa.csv", index=False)
print(r.groupby("op")[["greedy", "sa_best", "ratio", "t_lazy", "t_plain", "t_sa"]].mean().round(3))
