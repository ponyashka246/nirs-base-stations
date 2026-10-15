"""Независимая проверка модели покрытия по Speedtest by Ookla (мобильные тайлы, 4 кв. 2024)."""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

F = sys.argv[1]
cells = pd.read_parquet("out/cells.parquet"); cand = pd.read_parquet("out/cand.parquet")
pairs = pd.read_parquet("out/pairs_f1800.parquet")
t = pq.read_table(F, columns=["quadkey", "avg_d_kbps", "tests", "devices"]).to_pandas()


def qk_to_xyz(q):
    x = y = 0
    z = len(q)
    for i, ch in enumerate(q):
        m = 1 << (z - i - 1); d = int(ch)
        if d & 1: x |= m
        if d & 2: y |= m
    return x, y, z


xy = np.array([qk_to_xyz(q)[:2] for q in t.quadkey.values])
n = 2 ** 16
t["lon"] = (xy[:, 0] + 0.5) / n * 360 - 180
t["lat"] = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (xy[:, 1] + 0.5) / n))))
t["tx"], t["ty"] = xy[:, 0], xy[:, 1]
t = t[(t.lon.between(27.6, 35.9)) & (t.lat.between(58.3, 61.4))]
print("тайлов Ookla в рамке региона:", len(t))

# ячейки спроса -> тайл z16
cx = ((cells.lon.values + 180) / 360 * n).astype(int)
cy = ((1 - np.log(np.tan(np.radians(cells.lat.values)) + 1 / np.cos(np.radians(cells.lat.values))) / np.pi) / 2 * n).astype(int)
cells["tkey"] = cx.astype(np.int64) * n + cy
t["tkey"] = t.tx.astype(np.int64) * n + t.ty
ex = cand.kind.eq("existing").values
pe = pairs[ex[pairs.c.values]]
best = np.full(len(cells), np.inf); np.minimum.at(best, pe.b.values, pe.pl.values)
cells["best_pl"] = best
g = cells.groupby("tkey").agg(pop=("pop", "sum"), best_pl=("best_pl", "min")).reset_index()
m = g.merge(t[["tkey", "avg_d_kbps", "tests", "devices"]], on="tkey", how="left")
m["has"] = m.devices.fillna(0) >= 2
rows = {}
for th in (130, 135, 140):
    cov = m.best_pl <= th
    rows[th] = dict(recall=float(cov[m.has].mean()),
                    tested_share_cov=float(m.has[cov].mean()), tested_share_unc=float(m.has[~cov].mean()),
                    tests_per_1k_cov=float(m.tests[cov].fillna(0).sum() / m["pop"][cov].sum() * 1000),
                    tests_per_1k_unc=float(m.tests[~cov].fillna(0).sum() / m["pop"][~cov].sum() * 1000))
r = pd.DataFrame(rows).T
print(r.round(3))
h = m[m.has & np.isfinite(m.best_pl)]
rho = spearmanr(-h.best_pl, h.avg_d_kbps)
print(f"Спирмен: -потери vs скорость загрузки: rho={rho.correlation:.3f}, p={rho.pvalue:.1e}, n={len(h)}")
r.to_csv("out/ookla_check.csv")
pd.Series(dict(rho=rho.correlation, n=len(h), tiles=len(t), pop_tiles=len(g), tested_pop_tiles=int(m.has.sum()))).to_csv("out/ookla_rho.csv")
