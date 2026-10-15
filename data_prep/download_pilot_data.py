"""
Скачивает рельеф (Copernicus DEM GLO-30) и покрытие (ESA WorldCover 2021 v200)
для СПб и Ленинградской области (широта 58-61, долгота 27-35).
Только стандартная библиотека Python, ничего ставить не нужно.

Запуск: положить в папку "нирс" и запустить
    python download_pilot_data.py
или двойным щелчком по run_all.bat.
Файлы появятся рядом: data/dem и data/worldcover. Тайлов над морем не существует — они пропускаются.
"""
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEM_DIR = ROOT / "data" / "dem"
WC_DIR = ROOT / "data" / "worldcover"


def get(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print("уже есть ", dest.name)
        return
    try:
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dest)
        print(f"ok       {dest.name}  {dest.stat().st_size / 1e6:.1f} МБ")
    except urllib.error.HTTPError as e:
        print(f"нет      {dest.name}  (HTTP {e.code})")
    except Exception as e:
        print(f"ошибка   {dest.name}  ({e})")


def main():
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    WC_DIR.mkdir(parents=True, exist_ok=True)

    print("Рельеф Copernicus DEM GLO-30 (тайлы 1x1 градус):")
    for lat in range(58, 62):
        for lon in range(27, 36):
            name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            get(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif", DEM_DIR / f"{name}.tif")

    print("\nESA WorldCover 2021 v200 (тайлы 3x3 градуса):")
    for lat in (57, 60):
        for lon in (27, 30, 33):
            name = f"ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map"
            get(f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif", WC_DIR / f"{name}.tif")

    print("\nГотово.")


if __name__ == "__main__":
    main()
