"""Вариант C: прирост охвата при бюджете (колокация 1, новая вышка 2.5). CELF против жадного без учёта стоимости."""
import numpy as np, pandas as pd
from common import load, build, OPS
from opt import celf_budget, lazy_greedy
cells, cand, pairs = load("f1800")
rows=[]
for op in OPS:
    for var in ("A","B"):
        P, allowed, own = build(cells, cand, pairs, op, var, 135.0)
        for B in (25, 50, 125):
            S, f = celf_budget(P, B, allowed)
            # жадный без учёта стоимости, пока бюджет не кончится
            S0, _, _ = lazy_greedy(P, budget=B, ratio=False, allowed=allowed)
            Sr, _, _ = lazy_greedy(P, budget=B, ratio=True, allowed=allowed)
            rows.append(dict(op=op, variant=var, budget=B, celf=f, plain=P.value(S0), ratio=P.value(Sr), n_sites=len(S),
                             share_coloc=float((cand.kind.values[S]=="existing").mean())))
            print(rows[-1], flush=True)
r=pd.DataFrame(rows); r.to_csv("out/budget.csv", index=False)
r["gain_vs_plain"]=r.celf/r.plain
print(r.groupby(["variant","budget"])[["gain_vs_plain","n_sites","share_coloc"]].mean().round(3))
