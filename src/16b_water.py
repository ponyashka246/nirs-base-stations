"""Крупные водоёмы для подложки карты: класс 80 ESA WorldCover, огрублённый до ~300 м."""
import glob, json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import from_bounds
import shapely
from shapely.geometry import shape

polys = []
for f in sorted(glob.glob("data/wc/*.tif")):
    with rasterio.open(f) as s:
        n = 900
        a = s.read(1, out_shape=(n, n), resampling=Resampling.mode)
        b = s.bounds
        tr = from_bounds(b.left, b.bottom, b.right, b.top, n, n)
        m = (a == 80).astype(np.uint8)
        for g, v in shapes(m, mask=m.astype(bool), transform=tr):
            polys.append(shape(g))
u = shapely.union_all(polys).buffer(0)
geoms = list(u.geoms) if hasattr(u, "geoms") else [u]
geoms = [g for g in geoms if g.area > 0.002]
out = []
for g in geoms:
    g = g.simplify(0.003)
    for p in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
        if p.area < 0.002: continue
        out.append([[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords]] +
                   [[[round(x, 3), round(y, 3)] for x, y in r.coords] for r in p.interiors if shapely.Polygon(r).area > 0.0005])
print(len(out), sum(len(r) for p in out for r in p))
d = json.load(open("out/mapdata.json")); d["water"] = out
json.dump(d, open("out/mapdata.json", "w"), ensure_ascii=False, separators=(",", ":"))
