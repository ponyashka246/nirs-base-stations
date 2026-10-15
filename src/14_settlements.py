"""Кому достаётся покрытие: доля прироста в малых населённых пунктах (< 1000 жителей) и топ-10 мест для каждого оператора."""
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
from sklearn.neighbors import BallTree
from common import load, build, OPS
from opt import lazy_greedy

cells, cand, pairs = load("f1800")
pl = pyogrio.read_dataframe("data/osm.gpkg", layer="places")
pl = pl[pl.place.isin(["city", "town", "village", "hamlet", "isolated_dwelling"])].reset_index(drop=True)
tm = Transformer.from_crs(4326, 32636, always_xy=True)
px, py = tm.transform(pl.geometry.x.values, pl.geometry.y.values)
tree = BallTree(np.c_[px, py])
d, nn = tree.query(np.c_[cells.x.values, cells.y.values], k=1)
nn = nn[:, 0]; d = d[:, 0]
cells["place"] = np.where(d < 3000, nn, -1)
spb_city = pl.index[(pl.place == "city") & (pl.name == "Санкт-Петербург")]
size = cells[cells.place >= 0].groupby("place")["pop"].sum()
cells["set_pop"] = cells.place.map(size).fillna(0).values
cells.loc[cells.region.eq("Санкт-Петербург").values, "set_pop"] = 5e6
small = (cells.set_pop.values < 1000)
print(f"жителей в малых пунктах (<1000): {cells['pop'][small].sum():.0f} ({cells['pop'][small].sum()/cells['pop'].sum():.1%})")

rows, top = [], []
for op in OPS:
    P, allowed, own = build(cells, cand, pairs, op, "A", 135.0)
    unc = (P.v0 == 0)
    S, vals, cur = lazy_greedy(P, K=200, allowed=allowed)
    stand = P.standalone()
    order = allowed[np.argsort(-stand[allowed])]
    for K in (50, 200):
        def share(sel):
            c = P.v0.copy()
            for j in sel:
                P.add(j, c)
            g = P.w * (c - P.v0)
            return g[small].sum() / g.sum(), g[small].sum()
        sg, ng = share(S[:K]); sd, nd = share(order[:K])
        rows.append(dict(op=op, K=K, small_share_uncovered=P.w[unc & small].sum() / P.w[unc].sum(),
                         small_share_greedy=sg, small_people_greedy=ng, small_share_density=sd, small_people_density=nd))
        print(rows[-1], flush=True)
    prev = 0.0
    for r_, j in enumerate(S[:10]):
        g = vals[r_] - prev; prev = vals[r_]
        dd, ii = tree.query([[cand.x.values[j], cand.y.values[j]]], k=1)
        b = np.argmin((cells.x.values - cand.x.values[j]) ** 2 + (cells.y.values - cand.y.values[j]) ** 2)
        top.append(dict(op=op, rank=r_ + 1, gain=round(g), place=pl.name.values[ii[0, 0]], place_type=pl.place.values[ii[0, 0]],
                        dist_km=round(dd[0, 0] / 1000, 1), unit=cells.unit.values[b],
                        kind=("колокация (" + cand.ops.values[j] + ")") if cand.kind.values[j] == "existing" else "новая вышка",
                        lat=round(cand.lat.values[j], 5), lon=round(cand.lon.values[j], 5)))
pd.DataFrame(rows).to_csv("out/settlements.csv", index=False)
pd.DataFrame(top).to_csv("out/top10_sites.csv", index=False)
print(pd.DataFrame(top)[lambda x: x.op == "MegaFon"].to_string(index=False))
