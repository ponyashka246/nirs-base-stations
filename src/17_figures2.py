"""Дополнительные рисунки: схема конвейера, проверка спроса по МО, проверка покрытия по Ookla
(с рельефом и без), жадный алгоритм против точного решения, бюджетный вариант."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
OPN = {"MTS": "МТС", "Beeline": "Билайн", "MegaFon": "МегаФон", "T2": "Т2"}
OPC = {"MTS": BLUE, "Beeline": ORANGE, "MegaFon": AQUA, "T2": GRAY}

# ---------- схема конвейера ----------
fig, ax = plt.subplots(figsize=(10, 4.3)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 45)
BW, BH = 19, 7.6
ys = [33, 23.5, 14, 4.5]
cols = [
    ("Открытые данные", 1, ["Здания OSM\n(574 тыс. жилых)", "Петростат: численность\nрайонов и МО", "Реестр базовых\nстанций (32 984)", "Рельеф Copernicus DEM,\nпокров WorldCover"]),
    ("Модели (разд. 3)", 27, ["Модель спроса:\nжители каждого здания", None, "Сети четырёх\nоператоров", "Модель покрытия: Hata,\nпокров, дифракция"]),
    ("Оптимизация (разд. 2)", 53, ["Целевые функции:\nA, B, бюджет, вероятн.", None, "Ленивый жадный\nалгоритм и CELF", "Контроль: точная\nмодель ILP (HiGHS)"]),
    ("Результаты (разд. 4)", 79, ["Места для станций\nкаждого оператора", "Сравнение с базовыми\nметодами и сетями", "Проверка спроса\nпо 273 МО", "Проверка покрытия\nпо Ookla"]),
]
for title, x, items in cols:
    ax.text(x + BW / 2, 43.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    for y, t in zip(ys, items):
        if t is None:
            continue
        ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc="#e8f6f0" if x > 70 else "#eef4fc", ec=INK2, lw=0.8))
        ax.text(x + BW / 2, y + BH / 2, t, ha="center", va="center", fontsize=7.9, color=INK)
AR = dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=1.5)
LN = dict(color=INK2, lw=0.9)
mid = lambda y: y + BH / 2
def arr(x0, y0, x1, y1): ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=AR)
# данные -> модели
arr(20.4, mid(33), 26.6, mid(33)); arr(20.4, mid(23.5), 26.6, 34.5)
arr(20.4, mid(14), 26.6, mid(14)); arr(20.4, mid(4.5), 26.6, mid(4.5))
# модели -> шина -> целевые функции
bx = 49.5
for y in (33, 14, 4.5):
    ax.plot([46.4, bx], [mid(y), mid(y)], **LN)
ax.plot([bx, bx], [mid(4.5), mid(33)], **LN); arr(bx, mid(33), 52.6, mid(33))
# внутри оптимизации
arr(62.5, 32.6, 62.5, 22.0); arr(62.5, 13.6, 62.5, 12.5)
# оптимизация -> шина -> результаты
rx = 75.5
ax.plot([72.4, rx], [mid(14), mid(14)], **LN); ax.plot([rx, rx], [mid(4.5), mid(33)], **LN)
for y in ys:
    arr(rx, mid(y), 78.6, mid(y))
fig.savefig("out/fig0_pipeline.png"); plt.close(fig)

# ---------- проверка спроса по МО ----------
d = pd.read_csv("out/demand_check_mo_k0.6.csv")
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for reg, c, lab in [("Санкт-Петербург", BLUE, "МО Санкт-Петербурга"), ("Ленинградская область", ORANGE, "поселения Ленинградской области")]:
    s = d[d.region == reg]
    ax.scatter(s["true"], s["pred"], s=12, color=c, alpha=0.75, lw=0, label=f"{lab} ({len(s)})")
lim = [100, 3e5]
ax.plot(lim, lim, color=INK, lw=0.9)
ax.fill_between(lim, [v * 0.75 for v in lim], [v * 1.25 for v in lim], color=GRAY, alpha=0.15, lw=0, label="ошибка ±25 %")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Численность по данным Петростата на 01.01.2025, чел."); ax.set_ylabel("Оценка модели спроса, чел.")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig("out/fig5_demand_mo.png"); plt.close(fig)

# ---------- Ookla: по группам населения и абляция рельефа ----------
cells = pd.read_parquet("out/cells.parquet"); cand = pd.read_parquet("out/cand.parquet")
pairs = pd.read_parquet("out/pairs_f1800.parquet")
t = pq.read_table("data/ookla/2024-10-01_performance_mobile_tiles.parquet", columns=["quadkey", "devices"]).to_pandas()
q = np.array([list(s) for s in t.quadkey.values], dtype=np.int8) - ord("0")
w = 1 << np.arange(15, -1, -1)
tx = ((q & 1) * w).sum(1); ty = (((q >> 1) & 1) * w).sum(1)
n = 2 ** 16
t["tkey"] = tx.astype(np.int64) * n + ty
cx = ((cells.lon.values + 180) / 360 * n).astype(int)
cy = ((1 - np.log(np.tan(np.radians(cells.lat.values)) + 1 / np.cos(np.radians(cells.lat.values))) / np.pi) / 2 * n).astype(int)
cells["tkey"] = cx.astype(np.int64) * n + cy
ex = cand.kind.eq("existing").values
pe = pairs[ex[pairs.c.values]]
best = np.full(len(cells), np.inf); np.minimum.at(best, pe.b.values, pe.pl.values)
cells["best"] = best
g = cells.groupby("tkey").agg(pop=("pop", "sum"), best=("best", "min")).reset_index()
m = g.merge(t[["tkey", "devices"]], on="tkey", how="left")
m["has"] = m.devices.fillna(0) >= 2
cov = m.best <= 135
bins = [0, 30, 100, 300, 1000, 1e9]; labs = ["< 30", "30–100", "100–300", "300–1000", "> 1000"]
m["grp"] = pd.cut(m["pop"], bins, labels=labs, right=False)
r = m.groupby(["grp", cov], observed=True).has.mean().unstack()
print(r)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]; x = np.arange(len(labs)); bw = 0.38
ax.bar(x - bw / 2, r[True].values * 100, bw, color=BLUE, label="в зоне покрытия по модели")
ax.bar(x + bw / 2, r[False].values * 100, bw, color=ORANGE, label="вне зоны покрытия")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel("Жителей в тайле")
ax.set_ylabel("Доля тайлов с тестами Speedtest, %"); ax.legend(fontsize=8); ax.set_title("а) Где люди реально пользуются связью", fontsize=9.5, loc="left")
ax.grid(axis="x", visible=False)
ax = axes[1]
terr = dict(cov=[77.8, 83.8, 88.3], out=[141, 76, 39])
flat = dict(cov=[81.4, 87.5, 91.9], out=[115, 54, 23])
ax.plot(terr["cov"], terr["out"], "-o", color=BLUE, label="с рельефом")
ax.plot(flat["cov"], flat["out"], "-o", color=GRAY, label="без рельефа")
for (xx, yy, th) in zip(terr["cov"], terr["out"], (130, 135, 140)):
    ax.annotate(f"{th} дБ", (xx, yy), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=BLUE)
for (xx, yy, th) in zip(flat["cov"], flat["out"], (130, 135, 140)):
    ax.annotate(f"{th} дБ", (xx, yy), textcoords="offset points", xytext=(4, -11), fontsize=7.5, color=INK2)
ax.set_xlabel("Доля населённых тайлов, покрытых по модели, %"); ax.set_ylabel("Тайлов с тестами вне покрытия")
ax.legend(fontsize=8); ax.set_title("б) Учёт рельефа при равной доле покрытия", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig("out/fig6_ookla.png"); plt.close(fig)

# ---------- жадный против ILP ----------
sub = pd.read_csv("out/ilp_vs_greedy.csv")
fa = pd.read_csv("out/ilp_full.csv").assign(variant="A")
fb = pd.read_csv("out/ilp_full_b.csv")
full = pd.concat([fa[["op", "variant", "K", "ratio", "t_greedy", "t_ilp"]], fb[["op", "variant", "K", "ratio", "t_greedy", "t_ilp"]]])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
rng = np.random.default_rng(1)
groups = [("Подрегионы, A", sub[sub.variant == "A"].ratio), ("Подрегионы, B", sub[sub.variant == "B"].ratio),
          ("Весь регион, A", full[full.variant == "A"].ratio), ("Весь регион, B", full[full.variant == "B"].ratio)]
for i, (lab, v) in enumerate(groups):
    ax.scatter(np.clip(v.values, 0, 1) * 100, i + rng.uniform(-0.15, 0.15, len(v)), s=16, color=[BLUE, BLUE, AQUA, AQUA][i], alpha=0.8, lw=0)
ax.set_yticks(range(4)); ax.set_yticklabels([gl for gl, _ in groups]); ax.invert_yaxis()
ax.set_xlim(98, 100.1); ax.set_xlabel("Жадное решение, % от оптимума")
ax.set_title("а) Качество (гарантия в худшем случае — 63,2 %)", fontsize=9.5, loc="left"); ax.grid(axis="y", visible=False)
ax = axes[1]
for v, c in (("A", BLUE), ("B", ORANGE)):
    s = full[full.variant == v]
    ax.scatter(s.K + (-3 if v == "A" else 3), s.t_ilp, color=c, s=18, label=f"ILP, вариант {v}")
    ax.scatter(s.K + (-3 if v == "A" else 3), s.t_greedy, color=c, s=18, marker="^", label=f"жадный, вариант {v}")
ax.set_yscale("log"); ax.set_xlabel("Число новых станций K"); ax.set_ylabel("Время, с"); ax.set_xticks([50, 100, 200])
ax.legend(fontsize=7.5, ncol=2, loc="center right"); ax.set_title("б) Время на всём регионе", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig("out/fig7_ilp.png"); plt.close(fig)

# ---------- бюджет ----------
b = pd.read_csv("out/budget_sens.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for op in OPN:
    s = b[b.op == op]
    axes[0].plot(s.cost_new, s.gain * 100, "-o", color=OPC[op], label=OPN[op])
    axes[1].plot(s.cost_new, s.share_coloc * 100, "-o", color=OPC[op], label=OPN[op])
for ax in axes:
    ax.set_xticks([1.5, 2.5, 4]); ax.set_xlabel("Стоимость новой вышки (колокация = 1)")
axes[0].set_ylabel("Выигрыш CELF над жадным, %"); axes[1].set_ylabel("Доля колокаций в решении CELF, %")
axes[0].legend(fontsize=8); axes[0].set_title("а) Прирост охвата при том же бюджете", fontsize=9.5, loc="left")
axes[1].set_title("б) Из чего состоит решение", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig("out/fig8_budget.png"); plt.close(fig)
print("ok")
