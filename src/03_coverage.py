"""Модель покрытия: потери на трассе «кандидат -> H3-ячейка спроса».

PL = COST-231 Hata (f, hb, hm) с поправкой на тип местности по ESA WorldCover в точке приёма
     + потери дифракции на доминирующем препятствии рельефа (одиночный клин, ITU-R P.526).
Рельеф — Copernicus DEM (это DSM), приближённый к DTM минимумом по окну ~300 м.
Сохраняются пары с PL <= PL_KEEP (с запасом для анализа чувствительности по порогу).
"""
import glob
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import rasterio
from rasterio.merge import merge
from pyproj import Transformer
from sklearn.neighbors import BallTree

F_MHZ = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0
HB, HM = 30.0, 1.5
PL_KEEP = 145.0
RMAX_KM = 15.0
H3_DEMAND, H3_CAND = 9, 8
NPROF = 24
TAG = f"f{int(F_MHZ)}"

t0 = time.time()
to_m = Transformer.from_crs(4326, 32636, always_xy=True)

# ---------- спрос по ячейкам ----------
d = gpd.read_parquet("out/demand_k0.6.parquet")
d["h"] = [h3.latlng_to_cell(p.y, p.x, H3_DEMAND) for p in d.geometry]
cells = d.groupby("h").agg(pop=("pop", "sum"), region=("region", "first"), unit=("unit", "first")).reset_index()
ll = np.array([h3.cell_to_latlng(c) for c in cells.h])
cells["lat"], cells["lon"] = ll[:, 0], ll[:, 1]
cells["x"], cells["y"] = to_m.transform(cells.lon.values, cells.lat.values)
print(f"ячеек спроса: {len(cells)}, жителей {cells['pop'].sum():.0f}  [{time.time()-t0:.0f}s]")

# ---------- тип местности по WorldCover (мода по 9 точкам в ячейке) ----------
wc_files = sorted(glob.glob("data/wc/*.tif"))
wc_src = [rasterio.open(f) for f in wc_files]
offs = np.array([(dx, dy) for dx in (-0.0012, 0, 0.0012) for dy in (-0.0007, 0, 0.0007)])
cls = np.zeros((len(cells), len(offs)), dtype=np.uint8)
for s in wc_src:
    b = s.bounds
    for k, (dx, dy) in enumerate(offs):
        lon, lat = cells.lon.values + dx, cells.lat.values + dy
        m = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
        if m.any():
            vals = np.array([v[0] for v in s.sample(list(zip(lon[m], lat[m])))], dtype=np.uint8)
            cls[m, k] = vals
built = (cls == 50).sum(1)
tree = (cls == 10).sum(1)
cells["env"] = np.where(built >= 3, "urban", np.where(tree >= 5, "suburban", np.where(built >= 1, "suburban", "open")))
print("тип местности:", cells.env.value_counts().to_dict(), f"[{time.time()-t0:.0f}s]")

# ---------- DTM ----------
dem_files = sorted(glob.glob("data/dem/*.tif"))
RES = 0.0025  # ~ 280 м по широте, ~ 140 м по долготе
x0, y0, rx, ry = 27.0, 62.0, RES, -RES
dtm = np.zeros((int(4 / RES), int(9 / RES)), dtype=np.float32)
for f in dem_files:
    with rasterio.open(f) as s:
        a = s.read(1).astype(np.float32)
        a[a < -100] = 0
        fy, fx = int(round(RES / abs(s.res[1]))), int(round(RES / s.res[0]))
        h, w = a.shape[0] // fy * fy, a.shape[1] // fx * fx
        blk = a[:h, :w].reshape(h // fy, fy, w // fx, fx).min(axis=(1, 3))
        r0 = int(round((y0 - s.bounds.top) / RES)); c0 = int(round((s.bounds.left - x0) / RES))
        n = int(round(1 / RES))
        ri = (np.arange(n) * blk.shape[0] / n).astype(int); ci = (np.arange(n) * blk.shape[1] / n).astype(int)
        dtm[r0:r0 + n, c0:c0 + n] = blk[np.ix_(ri, ci)]


def ground(lon, lat):
    c = np.clip(((lon - x0) / rx).astype(int), 0, dtm.shape[1] - 1)
    r = np.clip(((lat - y0) / ry).astype(int), 0, dtm.shape[0] - 1)
    return dtm[r, c]


print(f"DTM {dtm.shape}, высоты {np.percentile(dtm,1):.0f}..{np.percentile(dtm,99):.0f} м  [{time.time()-t0:.0f}s]")

# ---------- кандидаты ----------
phys = pd.read_parquet("out/phys_sites.parquet")
tm = Transformer.from_crs(32636, 4326, always_xy=True)
phys["lon"], phys["lat"] = tm.transform(phys.x.values, phys.y.values)
cand_ex = pd.DataFrame({"cid": "E" + phys.phys.astype(str), "kind": "existing", "lon": phys.lon, "lat": phys.lat,
                        "ops": phys.ops, "n_ops": phys.n_ops})
pop8 = cells.assign(h8=[h3.cell_to_parent(c, H3_CAND) for c in cells.h]).groupby("h8")["pop"].sum()
pop8 = pop8[pop8 >= 20]
ll8 = np.array([h3.cell_to_latlng(c) for c in pop8.index])
cand_new = pd.DataFrame({"cid": "N" + pd.Series(pop8.index).astype(str), "kind": "new", "lon": ll8[:, 1], "lat": ll8[:, 0],
                         "ops": "", "n_ops": 0})
cand = pd.concat([cand_ex, cand_new], ignore_index=True)
cand["x"], cand["y"] = to_m.transform(cand.lon.values, cand.lat.values)
print(f"кандидатов: существующих {len(cand_ex)}, новых {len(cand_new)}  [{time.time()-t0:.0f}s]")

# ---------- модель потерь ----------
lf = np.log10(F_MHZ)
a_hm = (1.1 * lf - 0.7) * HM - (1.56 * lf - 0.8)
if F_MHZ >= 1500:
    A = 46.3 + 33.9 * lf - 13.82 * np.log10(HB) - a_hm
else:  # Okumura-Hata
    A = 69.55 + 26.16 * lf - 13.82 * np.log10(HB) - a_hm
Bs = 44.9 - 6.55 * np.log10(HB)
corr = {"urban": 0.0,
        "suburban": -2 * np.log10(F_MHZ / 28) ** 2 - 5.4,
        "open": -4.78 * lf ** 2 + 18.33 * lf - 40.94}
env_corr = cells.env.map(corr).values.astype(np.float32)
lam = 299.792458 / F_MHZ
KE = 4 / 3 * 6371000.0
cg = ground(cells.lon.values, cells.lat.values)


def knife(nu):
    out = np.zeros_like(nu)
    m = nu > -0.78
    v = nu[m] - 0.1
    out[m] = 6.9 + 20 * np.log10(np.sqrt(v * v + 1) + v)
    return out


tree_b = BallTree(np.c_[cells.x.values, cells.y.values])
idx_all = tree_b.query_radius(np.c_[cand.x.values, cand.y.values], r=RMAX_KM * 1000)
rows_c, rows_b, rows_pl = [], [], []
tt = np.linspace(0, 1, NPROF + 2)[1:-1]
for i, idx in enumerate(idx_all):
    if len(idx) == 0:
        continue
    dx = cells.x.values[idx] - cand.x.values[i]
    dy = cells.y.values[idx] - cand.y.values[i]
    dist = np.maximum(np.hypot(dx, dy), 20.0)
    pl = A + Bs * np.log10(dist / 1000) + env_corr[idx]
    keep = pl <= PL_KEEP
    if not keep.any():
        continue
    idx, dist, pl = idx[keep], dist[keep], pl[keep]
    # профиль рельефа
    lon0, lat0 = cand.lon.values[i], cand.lat.values[i]
    plon = lon0 + np.outer(cells.lon.values[idx] - lon0, tt)
    plat = lat0 + np.outer(cells.lat.values[idx] - lat0, tt)
    hz = ground(plon.ravel(), plat.ravel()).reshape(plon.shape)
    htx = ground(np.array([lon0]), np.array([lat0]))[0] + HB
    hrx = cg[idx] + HM
    d1 = np.outer(dist, tt)
    d2 = dist[:, None] - d1
    los = htx + (hrx - htx)[:, None] * tt[None, :]
    bulge = d1 * d2 / (2 * KE)
    hob = hz + bulge - los
    nu = hob * np.sqrt(2 * dist[:, None] / (lam * d1 * d2))
    jl = knife(nu.max(1))
    pl = pl + jl
    keep = pl <= PL_KEEP
    rows_c.append(np.full(keep.sum(), i, dtype=np.int32))
    rows_b.append(idx[keep].astype(np.int32))
    rows_pl.append(pl[keep].astype(np.float32))
    if i % 2000 == 0:
        print(f"   {i}/{len(cand)}  [{time.time()-t0:.0f}s]")

pairs = pd.DataFrame({"c": np.concatenate(rows_c), "b": np.concatenate(rows_b), "pl": np.concatenate(rows_pl)})
pairs.to_parquet(f"out/pairs_{TAG}.parquet")
cells.drop(columns=[]).to_parquet("out/cells.parquet")
cand.to_parquet("out/cand.parquet")
print(f"пар: {len(pairs):,}, среднее на кандидата {len(pairs)/len(cand):.0f}  [{time.time()-t0:.0f}s]")
for th in (130, 135, 140):
    cov = pairs[pairs.pl <= th].groupby("c").b.size()
    print(f"MAPL {th}: медианный охват кандидата {cov.median():.0f} ячеек")
