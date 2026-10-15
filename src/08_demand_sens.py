"""Чувствительность к модели спроса (коэффициент для частных домов k = 0.3 / 0.6 / 1.0)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from common import load, build, OPS
from opt import lazy_greedy

rows = []
base_sel = {}
for k in ["0.3", "0.6", "1.0"]:
    d = gpd.read_parquet(f"out/demand_k{k}.parquet")
    hh = pd.Series([h3.latlng_to_cell(p.y, p.x, 9) for p in d.geometry])
    dem = d["pop"].groupby(hh.values).sum()
    cells, cand, pairs = load("f1800", demand=dem)
    for op in OPS:
        for var in ("A", "B"):
            P, allowed, own = build(cells, cand, pairs, op, var, 135.0)
            S, vals, _ = lazy_greedy(P, K=50, allowed=allowed)
            stand = P.standalone()
            dens = P.value(allowed[np.argsort(-stand[allowed])[:50]])
            key = (op, var)
            if k == "0.6":
                base_sel[key] = set(S)
            rows.append(dict(k=k, op=op, variant=var, greedy=vals[-1], density=dens, sel=S))
r = pd.DataFrame(rows)
r["overlap_with_base"] = [len(set(s) & base_sel[(o, v)]) / 50 for s, o, v in zip(r.sel, r.op, r.variant)]
r.drop(columns="sel").to_csv("out/demand_sens.csv", index=False)
print(r.drop(columns="sel").groupby(["k", "variant"])[["greedy", "density", "overlap_with_base"]].mean().round(3))
