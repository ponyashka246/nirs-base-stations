import time, numpy as np, pandas as pd
from common import load, build, OPS
from opt import lazy_greedy, solve_ilp
cells, cand, pairs = load("f1800")
rows=[]
for op in OPS:
    for var,K in (("A",200),("B",50),("B",100)):
        P, allowed, own = build(cells, cand, pairs, op, var, 135.0)
        al = np.array([j for j in allowed if len(P.cc[j])])
        t0=time.time(); S, vals, _ = lazy_greedy(P, K=K, allowed=al); tg=time.time()-t0
        t0=time.time(); Si, fi, bound, st = solve_ilp(P, K, al, time_limit=900); ti=time.time()-t0
        rows.append(dict(op=op, variant=var, K=K, greedy=vals[-1], ilp=fi, status=str(st), ratio=vals[-1]/fi, t_greedy=tg, t_ilp=ti)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("out/ilp_full_b.csv", index=False)
