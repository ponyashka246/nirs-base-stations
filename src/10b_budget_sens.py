import numpy as np, pandas as pd, common
from opt import celf_budget, lazy_greedy
rows=[]
for cn in (1.5, 2.5, 4.0):
    common.COST_NEW = cn
    cells, cand, pairs = common.load("f1800")
    for op in common.OPS:
        P, allowed, own = common.build(cells, cand, pairs, op, "A", 135.0)
        S, f = celf_budget(P, 50, allowed)
        S0, _, _ = lazy_greedy(P, budget=50, ratio=False, allowed=allowed)
        rows.append(dict(cost_new=cn, op=op, celf=f, plain=P.value(S0), share_coloc=float((cand.kind.values[S]=="existing").mean())))
r=pd.DataFrame(rows); r["gain"]=r.celf/r.plain-1; r.to_csv("out/budget_sens.csv", index=False)
print(r.groupby("cost_new")[["gain","share_coloc"]].agg(["min","max"]).round(3))
