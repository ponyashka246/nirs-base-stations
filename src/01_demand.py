"""Модель спроса: оценка числа жителей в каждом здании СПб и ЛО.

w_b = A_b * L_b * k_type, затем нормировка внутри единицы калибровки к официальной численности.
A_b — площадь основания, L_b — этажность (тег, иначе высота/3, иначе медиана по типу и району),
k_type — относительная плотность проживания (многоквартирный дом 1.0, частный дом KHOUSE).
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio

KHOUSE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
SRC = "data/osm.gpkg"

# Официальная численность на 01.01.2025 (Петростат, по муниципальным образованиям), см. 00_population.py
import re as _re
_STOP = {"муниципальный", "муниципальное", "округ", "район", "городское", "сельское", "поселение", "поселок", "посёлок",
         "г", "гп", "город", "городской", "n", "№", "образование", "внутригородское"}
def key(s):
    s = s.lower().replace("ё", "е").replace("ѐ", "е")
    s = _re.sub(r"[.,«»\"()]", " ", s)
    return " ".join(w for w in s.split() if w not in _STOP)
OFF = pd.read_csv("out/districts_official.csv")
POP_TOTAL = OFF.groupby("region")["pop"].sum().to_dict()
MO = gpd.read_parquet("out/mo_official.parquet")

APART = {"apartments", "residential", "dormitory", "terrace", "yes"}
HOUSE = {"house", "detached", "semidetached_house", "bungalow", "farm", "cabin"}
NONRES_LU = {"industrial", "commercial", "retail", "garages", "military", "construction", "allotments"}

print("читаю здания...")
b = pyogrio.read_dataframe(SRC, layer="buildings")
adm = pyogrio.read_dataframe(SRC, layer="admin")
lu = pyogrio.read_dataframe(SRC, layer="landuse")

spb = adm[(adm.admin_level == "4") & (adm.name == "Санкт-Петербург")].geometry.iloc[0]
lo = adm[(adm.admin_level == "4") & (adm.name == "Ленинградская область")].geometry.iloc[0]
d5 = adm[adm.admin_level == "5"]
d5 = d5[d5.name.str.endswith("район") & d5.representative_point().within(spb)][["name", "geometry"]]
d6 = adm[adm.admin_level == "6"]
d6 = d6[d6.representative_point().within(lo)][["name", "geometry"]]
units = pd.concat([d5.assign(region="Санкт-Петербург"), d6.assign(region="Ленинградская область")])
units = gpd.GeoDataFrame(units, crs=4326).rename(columns={"name": "unit"})
units["unit"] = units.unit + np.where(units.region.eq("Санкт-Петербург"), " (СПб)", " (ЛО)")
print("единиц:", len(units))

b = gpd.sjoin(b, units, how="inner", predicate="within").drop(columns="index_right")
print("зданий в СПб+ЛО:", len(b))

# landuse под зданием
lu_j = gpd.sjoin(b[["geometry"]], lu[["landuse", "geometry"]], how="left", predicate="within")
lu_j = lu_j[~lu_j.index.duplicated()]
b["landuse"] = lu_j["landuse"].fillna("")

# жилое или нет
res_types = {"residential"}
is_res = b.object_type.isin(res_types)
unk = b.object_type.eq("unknown")
unk_res = unk & (b.landuse.eq("residential") | (~b.landuse.isin(NONRES_LU) & b.area_m2.between(30, 600)))
b["resid"] = is_res | unk_res
b = b[b.resid & (b.area_m2 >= 25)].copy()
print("жилых (с доопределением):", len(b), " из них по тегу:", int(is_res.loc[b.index].sum()))

# этажность
lv = pd.to_numeric(b.levels.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
h = pd.to_numeric(b.height.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
lv = lv.where(lv.between(1, 40))
lv = lv.fillna((h / 3.0).round().where((h / 3.0).between(1, 40)))
b["lv_src"] = np.where(lv.notna(), "tag", "imputed")
b["kind"] = np.where(b.building.isin(HOUSE), "house",
                     np.where(b.building.isin(APART - {"yes"}), "apart", "yes"))
# для building=yes: в городе (СПб) крупные — многоквартирные, мелкие — дома
big = b.area_m2 > 250
b.loc[b.kind.eq("yes") & big, "kind"] = "apart"
b.loc[b.kind.eq("yes") & ~big, "kind"] = "house"
b["lv"] = lv
med = b.groupby(["unit", "kind"]).lv.transform("median")
med2 = b.groupby(["kind"]).lv.transform("median")
b["lv"] = b.lv.fillna(med).fillna(med2).fillna(1).clip(1, 40)
b.loc[b.kind.eq("house"), "lv"] = b.loc[b.kind.eq("house"), "lv"].clip(upper=3)

b["k"] = np.where(b.kind.eq("house"), KHOUSE, 1.0)
b["raw"] = b.area_m2 * b.lv * b.k


b["ukey"] = b.region + "|" + b.unit.str.replace(r" \((СПб|ЛО)\)$", "", regex=True).map(key)
OFF["ukey"] = OFF.region + "|" + OFF["k"]
POP_UNIT = OFF.set_index("ukey")["pop"].to_dict()
miss = set(b.ukey.unique()) - set(POP_UNIT)
print("единиц без официальной численности:", miss)


def calibrate(df, level="unit"):
    pop = pd.Series(0.0, index=df.index)
    if level == "region":
        for reg, tot in POP_TOTAL.items():
            m = df.region.eq(reg)
            pop[m] = df.raw[m] / df.raw[m].sum() * tot
    else:
        s = df.raw.groupby(df.ukey).transform("sum")
        pop = df.raw / s * df.ukey.map(POP_UNIT)
    return pop


# Проверка 1: калибровка только по итогу региона -> ошибка по 36 районам
p1 = calibrate(b, "region")
chk = pd.DataFrame({"pred": p1.groupby(b.ukey).sum()}).join(pd.Series(POP_UNIT, name="true"), how="inner")
chk["err"] = (chk.pred - chk.true) / chk.true
chk["region"] = chk.index.str.split("|").str[0]
print("\nПроверка 1 (калибровка по региону, ошибка по районам):")
print(chk.round(3).to_string())
print("MAPE по районам:", chk.groupby("region").err.apply(lambda e: e.abs().mean()).round(3).to_dict(), "всего", round(chk.err.abs().mean(), 3))

# Проверка 2: калибровка по районам -> ошибка по муниципальным образованиям (МО СПб, поселения ЛО)
p2 = calibrate(b, "unit")
bb = gpd.GeoDataFrame({"pop": p2.values}, geometry=b.geometry.values, crs=4326)
j = gpd.sjoin(bb, MO[["name", "region", "pop", "geometry"]].rename(columns={"pop": "true"}), how="inner", predicate="within")
mo = j.groupby(["region", "name"]).agg(pred=("pop", "sum"), true=("true", "first")).reset_index()
mo["err"] = (mo.pred - mo.true) / mo.true
mo["size"] = pd.cut(mo.true, [0, 1000, 5000, 20000, 1e9], labels=["<1 тыс.", "1-5 тыс.", "5-20 тыс.", ">20 тыс."])
print("\nПроверка 2 (калибровка по районам, ошибка по МО):", len(mo), "МО")
print(mo.groupby("region").err.apply(lambda e: e.abs().median()).round(3).to_dict(), "медиана |ошибки|")
print(mo.groupby("region").err.apply(lambda e: e.abs().mean()).round(3).to_dict(), "средняя |ошибка|")
print(mo.groupby("size", observed=True).err.apply(lambda e: e.abs().median()).round(3).to_dict(), "медиана по размеру")
wape = (mo.pred - mo.true).abs().sum() / mo.true.sum()
print("WAPE по МО:", round(wape, 3))
mo.to_csv(f"out/demand_check_mo_k{KHOUSE}.csv", index=False)

b["pop"] = p2
out = b[["osm_id", "unit", "region", "kind", "lv", "lv_src", "area_m2", "pop", "geometry"]]
out.to_parquet(f"out/demand_k{KHOUSE}.parquet")
chk.to_csv(f"out/demand_check_k{KHOUSE}.csv")
print("\nжителей всего:", round(out["pop"].sum()))
print(out.groupby("region")["pop"].sum().round())
print(out.groupby("kind")["pop"].sum().round())
print("этажность из тега, доля жителей:", round(out["pop"][out.lv_src.eq('tag')].sum() / out["pop"].sum(), 3))
