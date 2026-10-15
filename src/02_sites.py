"""Вышки -> площадки. Записи реестра одного оператора в радиусе R склеиваются в одну площадку оператора;
записи всех операторов в радиусе R — в одну физическую площадку (для колокации)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph

R = 50.0
OPS = {"МТС": "MTS", "Билайн": "Beeline", "Теле2": "T2", "Мегафон": "MegaFon"}
NORM = {"Теле2*": "Теле2", "MTC": "МТС", "МТС*": "МТС", "Мегафон*": "Мегафон"}

df = pd.read_csv("data/towers.csv", sep=";", encoding="utf-8-sig", dtype=str)
df["lat"] = df.lat.astype(float)
df["lng"] = df.lng.astype(float)
df["date"] = pd.to_datetime(df.date, errors="coerce")
v = df.vendor.replace(NORM).str.split(",").str[0].str.strip()
df["op"] = v.map(OPS)
n0 = len(df)
df = df[df.op.notna()].copy()

adm = pyogrio.read_dataframe("data/osm.gpkg", layer="admin")
reg = adm[(adm.admin_level == "4") & adm.name.isin(["Санкт-Петербург", "Ленинградская область"])].union_all()
g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs=4326)
g = g[g.within(reg)].reset_index(drop=True)
print(f"записей: {n0}, большой четвёрки в СПб+ЛО: {len(g)}")

xy = g.to_crs(32636)
X = np.c_[xy.geometry.x, xy.geometry.y]


def cluster(idx):
    A = radius_neighbors_graph(X[idx], R, mode="connectivity")
    return connected_components(A, directed=False)[1]


lab = cluster(np.arange(len(g)))
g["phys"] = lab
g["opsite"] = ""
for op in OPS.values():
    m = np.where(g.op.eq(op))[0]
    g.loc[m, "opsite"] = op + "_" + pd.Series(cluster(m)).astype(str).values

g["x"], g["y"] = X[:, 0], X[:, 1]
sites = g.groupby("opsite").agg(op=("op", "first"), phys=("phys", "first"), x=("x", "mean"), y=("y", "mean"),
                                 n_rec=("id", "size"), first=("date", "min"), last=("date", "max")).reset_index()
phys = g.groupby("phys").agg(x=("x", "mean"), y=("y", "mean"), ops=("op", lambda s: ",".join(sorted(set(s))))).reset_index()
phys["n_ops"] = phys.ops.str.count(",") + 1
sites.to_parquet("out/op_sites.parquet")
phys.to_parquet("out/phys_sites.parquet")
print("площадок операторов:", sites.op.value_counts().to_dict())
print("физических площадок:", len(phys), " по числу операторов:", phys.n_ops.value_counts().sort_index().to_dict())
print("площадок, где последняя запись до 2015:", int((sites["last"] < "2015-01-01").sum()))
