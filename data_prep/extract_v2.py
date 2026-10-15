"""
Повторная выгрузка из .osm.pbf для СПб и Ленинградской области, v2.
Отличия от extract.py: сохраняет этажность и высоту, берёт и мультиполигоны (relations),
пишет здания точками (центроид + площадь), чтобы файл был маленьким,
и дополнительно выгружает landuse, населённые пункты с населением и административные границы.

Запуск из папки "полигоны" с тем же venv:
    venv\\Scripts\\python extract_v2.py
Результат: output/spb_lo_v2.gpkg (слои buildings, landuse, places, admin).
"""
import math
import time
from pathlib import Path

import osmium
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point

from classifier import classify_osm_feature

PBF = "northwestern-fed-district-260831.osm.pbf"
OUT = Path("output/spb_lo_v2.gpkg")
BBOX = (27.6, 58.3, 35.9, 61.4)  # lon_min, lat_min, lon_max, lat_max (СПб + ЛО с запасом)

LANDUSE_KEEP = {"residential", "industrial", "commercial", "retail", "allotments",
                "garages", "farmyard", "military", "construction"}
PLACE_KEEP = {"city", "town", "village", "hamlet", "isolated_dwelling", "suburb",
              "quarter", "neighbourhood", "locality"}
ADMIN_KEEP = {"4", "5", "6", "8"}

M_PER_DEG = 111320.0


def in_bbox(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


def area_m2(geom, lat):
    # площадь в градусах -> м2 в локальной равнопромежуточной проекции (ошибка < 0.5%)
    return geom.area * M_PER_DEG * M_PER_DEG * math.cos(math.radians(lat))


class Handler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.wkbf = osmium.geom.WKBFactory()
        self.buildings, self.landuse, self.places, self.admin = [], [], [], []
        self.errors = 0

    def node(self, n):
        place = n.tags.get("place")
        if place in PLACE_KEEP and n.location.valid():
            lon, lat = n.location.lon, n.location.lat
            if in_bbox(lon, lat):
                self.places.append({
                    "osm_id": n.id, "place": place,
                    "name": n.tags.get("name", ""),
                    "population": n.tags.get("population", ""),
                    "population_date": n.tags.get("population:date", ""),
                    "oktmo": n.tags.get("oktmo:user", "") or n.tags.get("ref:oktmo", ""),
                    "geometry": Point(lon, lat),
                })

    def area(self, a):
        tags = a.tags
        is_building = "building" in tags
        lu = tags.get("landuse")
        is_landuse = lu in LANDUSE_KEEP
        is_admin = tags.get("boundary") == "administrative" and tags.get("admin_level") in ADMIN_KEEP
        if not (is_building or is_landuse or is_admin):
            return
        try:
            geom = wkb.loads(self.wkbf.create_multipolygon(a), hex=True)
        except Exception:
            self.errors += 1
            return
        if geom.is_empty:
            return
        minx, miny, maxx, maxy = geom.bounds
        if maxx < BBOX[0] or minx > BBOX[2] or maxy < BBOX[1] or miny > BBOX[3]:
            return
        osm_id = a.orig_id()
        osm_type = "w" if a.from_way() else "r"

        if is_building:
            c = geom.representative_point()
            if not in_bbox(c.x, c.y):
                return
            t = dict(tags)
            self.buildings.append({
                "osm_id": osm_id, "osm_type": osm_type,
                "building": tags.get("building", ""),
                "levels": tags.get("building:levels", ""),
                "height": tags.get("height", "") or tags.get("building:height", ""),
                "flats": tags.get("building:flats", ""),
                "object_type": classify_osm_feature(t),
                "area_m2": round(area_m2(geom, c.y), 1),
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "geometry": c,
            })
            if len(self.buildings) % 100000 == 0:
                print(f"   ... зданий: {len(self.buildings):,}")
        elif is_landuse:
            self.landuse.append({"osm_id": osm_id, "osm_type": osm_type, "landuse": lu,
                                 "geometry": geom})
        if is_admin:
            self.admin.append({"osm_id": osm_id, "osm_type": osm_type,
                               "admin_level": tags.get("admin_level"),
                               "name": tags.get("name", ""),
                               "oktmo": tags.get("oktmo:user", "") or tags.get("ref:oktmo", ""),
                               "geometry": geom})


def main():
    if not Path(PBF).exists():
        print(f"Нет файла {PBF} в папке {Path.cwd()}")
        return
    t0 = time.time()
    h = Handler()
    print("Читаю PBF (несколько минут)...")
    h.apply_file(PBF, locations=True, idx="flex_mem")
    print(f"Готово за {time.time() - t0:.0f} с. Ошибок геометрии: {h.errors}")

    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    for name, rows in [("buildings", h.buildings), ("landuse", h.landuse),
                       ("places", h.places), ("admin", h.admin)]:
        if not rows:
            print(f"   {name}: 0 (слой пропущен)")
            continue
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(OUT, layer=name, driver="GPKG")
        print(f"   {name}: {len(gdf):,}")
    print(f"Файл: {OUT}  ({OUT.stat().st_size / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
