"""Точное решение (HiGHS) против жадного на подрегионах: реальный разрыв с оптимумом и оценка Лесковца."""
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from common import load, build
from opt import lazy_greedy, online_bound, solve_ilp

cells, cand, pairs = load("f1800")
d = gpd.read_parquet("out/demand_k0.6.parquet")
units = ["Лужский район (ЛО)", "Приозерский район (ЛО)", "Тихвинский район (ЛО)", "Волосовский район (ЛО)",
         "Кировский район (ЛО)", "Ломоносовский район (ЛО)"]
cand_unit = gpd.sjoin(gpd.GeoDataFrame(cand, geometry=gpd.points_from_xy(cand.lon, cand.lat), crs=4326),
                      d.dissolve("unit").reset_index()[["unit", "geometry"]].assign(
                          geometry=lambda x: x.geometry.convex_hull), how="left", predicate="within")
cand_unit = cand_unit[~cand_unit.index.duplicated()]["unit"].values
rows = []
t0 = time.time()
for op in ["T2", "MTS"]:
    for var in ["A", "B"]:
        P, allowed, own = build(cells, cand, pairs, op, var, 135.0)
        w_full = P.w.copy()
        for u in units:
            cm = cells.unit.eq(u).values
            P.w = np.where(cm, w_full, 0.0)
            al = np.array([j for j in allowed if cand_unit[j] == u and len(P.cc[j]) and P.w[P.cc[j]].sum() > 0])
            for K in [5, 10, 20]:
                if len(al) <= K:
                    continue
                S, vals, _ = lazy_greedy(P, K=K, allowed=al)
                g = vals[-1] if vals else 0.0
                _, ub = online_bound(P, S, K, al)
                Si, fi, bound, st = solve_ilp(P, K, al, time_limit=300)
                rows.append(dict(op=op, variant=var, unit=u, K=K, n_cand=len(al), greedy=g, ilp=fi, ilp_bound=bound,
                                 status=str(st), ratio=g / fi if fi > 0 else np.nan, online_ub=ub))
                print(f"{op} {var} {u:28s} K={K:2d} cand={len(al):4d} greedy={g:8.0f} ilp={fi:8.0f} "
                      f"ratio={g/max(fi,1e-9):.4f} online_ub={ub:8.0f} {st}  [{time.time()-t0:.0f}s]", flush=True)
        P.w = w_full
pd.DataFrame(rows).to_csv("out/ilp_vs_greedy.csv", index=False)
r = pd.DataFrame(rows)
print(r.groupby(["variant", "K"]).ratio.agg(["mean", "min"]).round(4))
