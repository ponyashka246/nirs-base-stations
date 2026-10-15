"""Данные для интерактивной карты: рекомендованные места (жадный алгоритм, варианты A и B, K до 100),
площадки операторов, непокрытое население по шестиугольникам H3 res 7, границы районов."""
import json
import numpy as np
import pandas as pd
import pyogrio
import h3
import shapely
from sklearn.neighbors import BallTree
from pyproj import Transformer
from common import load, build, OPS
from opt import lazy_greedy

K = 100
TH = 135.0
cells, cand, pairs = load("f1800")
out = {"ops": {}, "meta": dict(K=K, mapl=TH, mhz=1800)}

# населённые пункты — для подписи мест
pl = pyogrio.read_dataframe("data/osm.gpkg", layer="places")
pl = pl[pl.place.isin(["city", "town", "village", "hamlet", "isolated_dwelling"])].reset_index(drop=True)
tm = Transformer.from_crs(4326, 32636, always_xy=True)
px, py = tm.transform(pl.geometry.x.values, pl.geometry.y.values)
tree = BallTree(np.c_[px, py])

# шестиугольники res 7
cells["h7"] = [h3.cell_to_parent(c, 7) for c in cells.h]
h7 = cells.groupby("h7")["pop"].sum()
h7 = h7[h7 >= 1]
h7_index = {h: i for i, h in enumerate(h7.index)}
out["hex"] = [[[round(lng, 4), round(lat, 4)] for lat, lng in h3.cell_to_boundary(h)] for h in h7.index]
out["hex_pop"] = [round(float(v)) for v in h7.values]
hidx = cells.h7.map(h7_index).fillna(-1).astype(int).values

for op in OPS:
    d = {}
    own_mask = cand.ops.str.contains(op).values & cand.kind.eq("existing").values
    d["sites"] = np.round(np.c_[cand.lon.values[own_mask], cand.lat.values[own_mask]], 4).tolist()
    for var in ("A", "B"):
        P, allowed, own = build(cells, cand, pairs, op, var, TH)
        if var == "A":
            unc = P.w * (1 - P.v0)
            d["covered_share"] = float((P.w * P.v0).sum() / P.w.sum())
            hv = np.zeros(len(h7)); m = hidx >= 0
            np.add.at(hv, hidx[m], unc[m])
            d["hex_unc"] = [round(float(v)) for v in hv]
        else:
            d["captured_base"] = round(float((P.w * P.v0).sum()))
        S, vals, _ = lazy_greedy(P, K=K, allowed=allowed)
        rows, prev = [], 0.0
        for r, j in enumerate(S[:K]):
            g = vals[r] - prev; prev = vals[r]
            dd, ii = tree.query([[cand.x.values[j], cand.y.values[j]]], k=1)
            b = np.argmin((cells.x.values - cand.x.values[j]) ** 2 + (cells.y.values - cand.y.values[j]) ** 2)
            rows.append(dict(r=r + 1, g=round(float(g)), lon=round(float(cand.lon.values[j]), 5), lat=round(float(cand.lat.values[j]), 5),
                             k="c" if cand.kind.values[j] == "existing" else "n", on=cand.ops.values[j],
                             p=pl.name.values[ii[0, 0]], pd=round(float(dd[0, 0]) / 1000, 1), u=cells.unit.values[b]))
        d[var] = rows
        print(op, var, "сумма", round(prev), "колокаций", sum(x["k"] == "c" for x in rows), flush=True)
    out["ops"][op] = d

# границы
ad = pyogrio.read_dataframe("data/osm.gpkg", layer="admin")
reg = ad[ad.name.isin(["Санкт-Петербург", "Ленинградская область"])]
spb = reg[reg.name == "Санкт-Петербург"].geometry.values[0]
lo = reg[reg.name == "Ленинградская область"].geometry.values[0]
both = shapely.union(spb, lo)


def coords(geom, tol):
    g = shapely.simplify(geom, tol, preserve_topology=True)
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms) if hasattr(g, "geoms") else []
    res = []
    for p in polys:
        if p.geom_type != "Polygon" or p.area < tol * tol * 4:
            continue
        res.append([[round(x, 3), round(y, 3)] for x, y in p.exterior.coords])
    return res


out["region"] = {"СПб": coords(spb, 0.002), "ЛО": coords(lo, 0.003)}
dist = ad[(ad.admin_level == "5") & ad.geometry.representative_point().within(spb)
          | (ad.admin_level == "6") & ad.geometry.representative_point().within(lo)]
out["districts"] = [dict(n=r.name, g=coords(r.geometry, 0.003)) for r in dist.itertuples() if r.name]
nb = ad[(ad.admin_level == "4") & ~ad.name.isin(["Санкт-Петербург", "Ленинградская область", ""])]
bbox = shapely.box(27.0, 58.2, 36.2, 61.9)
out["neighbors"] = [coords(shapely.intersection(g, bbox), 0.01) for g in nb.geometry.values]
big = pl[pl.place.isin(["city", "town"])]
big = big[big.geometry.within(both)]
out["towns"] = [dict(n=r.name, lon=round(r.geometry.x, 3), lat=round(r.geometry.y, 3), c=r.place == "city") for r in big.itertuples()]
json.dump(out, open("out/mapdata.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("towns", len(out["towns"]), "districts", len(out["districts"]), "hex", len(out["hex"]))
