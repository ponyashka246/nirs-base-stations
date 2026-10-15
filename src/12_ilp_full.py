"""Точное решение на всём регионе (СПб + ЛО), вариант A, K = 50 и 100."""
import time
import numpy as np
import pandas as pd
from common import load, build, OPS
from opt import lazy_greedy, online_bound, solve_ilp

cells, cand, pairs = load("f1800")
rows = []
for op in OPS:
    P, allowed, own = build(cells, cand, pairs, op, "A", 135.0)
    al = np.array([j for j in allowed if len(P.cc[j])])
    for K in (50, 100):
        t0 = time.time(); S, vals, _ = lazy_greedy(P, K=K, allowed=al); tg = time.time() - t0
        _, ub = online_bound(P, S, K, al)
        t0 = time.time(); Si, fi, bound, st = solve_ilp(P, K, al, time_limit=900); ti = time.time() - t0
        rows.append(dict(op=op, K=K, n_cand=len(al), greedy=vals[-1], ilp=fi, ilp_bound=bound, status=str(st),
                         ratio=vals[-1] / fi, ratio_bound=vals[-1] / bound, online_ub=ub, t_greedy=tg, t_ilp=ti))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("out/ilp_full.csv", index=False)
