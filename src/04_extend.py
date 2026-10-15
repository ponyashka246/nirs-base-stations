"""Протокол «достройка»: оператор добавляет K станций к своей сети. Сравнение алгоритмов и бейзлайнов."""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from common import load, build, OPS
from opt import lazy_greedy, online_bound, swap_search

TH = float(sys.argv[1]) if len(sys.argv) > 1 else 135.0
TAG = sys.argv[2] if len(sys.argv) > 2 else "f1800"
KS = [10, 25, 50, 100, 200]
rng = np.random.default_rng(42)
cells, cand, pairs = load(TAG)
rows = []
t0 = time.time()
for op in OPS:
    for var in ("A", "B"):
        P, allowed, own = build(cells, cand, pairs, op, var, TH)
        stand = P.standalone()
        S, vals, _ = lazy_greedy(P, K=max(KS), allowed=allowed)
        # k-means по ещё не обслуженному спросу
        unserved = P.w * (1 - P.v0)
        m = unserved > 0
        X = np.c_[cells.x.values[m], cells.y.values[m]]
        cx, cy = cand.x.values[allowed], cand.y.values[allowed]
        for K in KS:
            g = vals[K - 1] if len(vals) >= K else vals[-1]
            fS, ub = online_bound(P, S[:K], K, allowed)
            rnd = np.mean([P.value(rng.choice(allowed, K, replace=False)) for _ in range(20)])
            top = allowed[np.argsort(-stand[allowed])[:K]]
            dens = P.value(top)
            km = KMeans(K, n_init=3, random_state=0).fit(X, sample_weight=unserved[m])
            near = [allowed[np.argmin((cx - a) ** 2 + (cy - b) ** 2)] for a, b in km.cluster_centers_]
            kmv = P.value(list(dict.fromkeys(near)))
            sw = np.nan
            if K <= 50:
                _, sw = swap_search(P, S[:K], allowed, max_rounds=1, sample=300)
            rows.append(dict(op=op, variant=var, K=K, greedy=g, greedy_swap=sw, ub=ub, gap_ub=1 - g / ub,
                             random=rnd, density=dens, kmeans=kmv, base=float((P.w * P.v0).sum())))
            print(f"{op:8s} {var} K={K:3d} greedy={g:9.0f} swap={sw:9.0f} ub={ub:9.0f} gap<={1-g/ub:.3f} "
                  f"rnd={rnd:8.0f} dens={dens:8.0f} km={kmv:8.0f}  [{time.time()-t0:.0f}s]", flush=True)
        n_coloc = int((cand.kind.values[S[:50]] == "existing").sum())
        print(f"   {op} {var}: из 50 жадных мест на чужих площадках (колокация) {n_coloc}")
pd.DataFrame(rows).to_csv(f"out/extend_th{int(TH)}_{TAG}.csv", index=False)
