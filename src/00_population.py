"""Разбор таблиц Петростата (численность на 01.01.2025 по МО) и сопоставление с границами OSM."""
import re
import subprocess
import pandas as pd
import pyogrio

SRC = "data/population/"  # PDF Петростата с численностью по МО
STOP = {"муниципальный", "муниципальное", "округ", "район", "городское", "сельское", "поселение", "поселок", "посёлок",
        "г", "гп", "город", "городской", "n", "№", "образование", "внутригородское"}


def key(s):
    s = s.lower().replace("ё", "е").replace("ѐ", "е")
    s = re.sub(r"[.,«»\"()]", " ", s)
    return " ".join(w for w in s.split() if w not in STOP)


def rows(pdf):
    txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    out = []
    for ln in txt.splitlines():
        m = re.match(r"^(\s*)(\S.*?\S)\s{2,}(\d{2,8})\b", ln)
        if m and not m.group(2).startswith(("Численность", "Человек")):
            out.append((len(m.group(1)), m.group(2).strip(), int(m.group(3))))
    return out


dist, mo = [], []
cur = None
for ind, name, val in rows(SRC + "СПб числ на 01.01.2025 по МО.pdf"):
    if name == "Санкт-Петербург":
        continue
    if name.endswith("район"):
        cur = name; dist.append(("Санкт-Петербург", name, val))
    elif cur:
        mo.append(("Санкт-Петербург", cur, name, val))
cur = None
for ind, name, val in rows(SRC + "ЛО числ на 01.01.2025 по МО.pdf"):
    if name == "Ленинградская область":
        continue
    if re.search(r"(муниципальный район|муниципальный округ|городской округ)$", name):
        cur = name; dist.append(("Ленинградская область", name, val))
    elif name.startswith(("Городское поселение", "Сельское поселение")) and cur:
        mo.append(("Ленинградская область", cur, name, val))
D = pd.DataFrame(dist, columns=["region", "name", "pop"])
M = pd.DataFrame(mo, columns=["region", "district", "name", "pop"])
print("районов:", len(D), D.groupby("region")["pop"].sum().to_dict())
print("МО:", len(M), M.groupby("region")["pop"].sum().to_dict())

adm = pyogrio.read_dataframe("data/osm.gpkg", layer="admin")
spb = adm[(adm.admin_level == "4") & (adm.name == "Санкт-Петербург")].geometry.iloc[0]
lo = adm[(adm.admin_level == "4") & (adm.name == "Ленинградская область")].geometry.iloc[0]
a8 = adm[adm.admin_level == "8"].copy()
rp = a8.representative_point()
a8["region"] = ["Санкт-Петербург" if spb.contains(p) else ("Ленинградская область" if lo.contains(p) else None) for p in rp]
a8 = a8[a8.region.notna()]
a8["k"] = a8.name.map(key)
M["k"] = M.name.map(key)
m = M.merge(a8[["region", "k", "osm_id", "geometry"]], on=["region", "k"], how="left")
dup = m.k.duplicated(keep=False) & m.osm_id.notna()
print("сопоставлено МО:", m.osm_id.notna().sum(), "из", len(M), " неоднозначных:", int(dup.sum()))
print("не найдены:", m[m.osm_id.isna()].name.head(30).tolist())
m = m[m.osm_id.notna() & ~m.duplicated(["region", "k"], keep=False)]
import geopandas as gpd
gpd.GeoDataFrame(m, geometry="geometry", crs=4326).to_parquet("out/mo_official.parquet")
D["k"] = D.name.map(key)
D.to_csv("out/districts_official.csv", index=False)
