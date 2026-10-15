"""Рисунки для статьи (PNG, 200 dpi, светлая тема для печати)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from common import load, build, OPS
from opt import lazy_greedy, online_bound
from sklearn.cluster import KMeans

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
OPNAME = {"MTS": "МТС", "Beeline": "Билайн", "MegaFon": "МегаФон", "T2": "Т2"}

cells, cand, pairs = load("f1800")
adm = pyogrio.read_dataframe("data/osm.gpkg", layer="admin")
reg = adm[(adm.admin_level == "4") & adm.name.isin(["Санкт-Петербург", "Ленинградская область"])].to_crs(32636)
dist = adm[adm.admin_level.isin(["5", "6"])].to_crs(32636)
dist = dist[dist.representative_point().within(reg.union_all())]


def basemap(ax, bbox=None):
    dist.boundary.plot(ax=ax, color="#c9c8c3", linewidth=0.3)
    reg.boundary.plot(ax=ax, color=INK2, linewidth=0.7)
    ax.set_axis_off()
    if bbox:
        ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")


# ---------- Рис. 1: спрос и белые пятна ----------
p = pairs[pairs.pl <= 135]
ex = np.where(cand.kind.eq("existing").values)[0]
cov_any = np.zeros(len(cells), bool); cov_any[p[np.isin(p.c.values, ex)].b.values] = True
dens = cells["pop"].values / 0.105  # чел/км2 (площадь ячейки H3 res 9 ~0,105 км2)
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
for ax, bbox in zip(axes, [None, (325000, 6620000, 390000, 6685000)]):
    basemap(ax, bbox)
    m = cells["pop"].values > 0
    sc = ax.scatter(cells.x[m], cells.y[m], c=dens[m], s=0.6 if bbox is None else 1.2, cmap=SEQ,
                    norm=LogNorm(10, 30000), linewidths=0, rasterized=True)
    u = ~cov_any & m
    ax.scatter(cells.x[u], cells.y[u], s=1.0 if bbox is None else 4, color=ORANGE, linewidths=0, rasterized=True)
axes[0].set_title("Ленинградская область и Санкт-Петербург", fontsize=9, color=INK)
axes[1].set_title("Санкт-Петербург", fontsize=9, color=INK)
cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01)
cb.set_label("плотность населения, чел./км² (лог. шкала)")
axes[0].scatter([], [], color=ORANGE, s=12, label=f"не покрыто ни одним оператором ({(cells['pop'][u].sum()/1e3):.0f} тыс. чел.)")
axes[0].legend(loc="lower left", fontsize=8)
fig.savefig("out/fig1_demand_gaps.png"); plt.close(fig)
print("рис.1: непокрыто", round(cells["pop"][~cov_any].sum()))

# ---------- Рис. 2 и 3: МегаФон, достройка ----------
OP = "MegaFon"
res = {}
for var in ("A", "B"):
    P, allowed, own = build(cells, cand, pairs, OP, var, 135.0)
    S, vals, _ = lazy_greedy(P, K=200, allowed=allowed)
    stand = P.standalone()
    order = allowed[np.argsort(-stand[allowed])]
    unserved = P.w * (1 - P.v0); mk = unserved > 0
    X = np.c_[cells.x.values[mk], cells.y.values[mk]]
    cx, cy = cand.x.values[allowed], cand.y.values[allowed]
    Ks = [1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200]
    curves = {"greedy": [vals[k - 1] for k in Ks], "density": [P.value(order[:k]) for k in Ks], "kmeans": [], "random": [], "ub": []}
    rng = np.random.default_rng(1)
    for k in Ks:
        km = KMeans(k, n_init=3, random_state=0).fit(X, sample_weight=unserved[mk])
        near = [allowed[np.argmin((cx - a) ** 2 + (cy - b) ** 2)] for a, b in km.cluster_centers_]
        curves["kmeans"].append(P.value(list(dict.fromkeys(near))))
        curves["random"].append(np.mean([P.value(rng.choice(allowed, k, replace=False)) for _ in range(20)]))
        curves["ub"].append(online_bound(P, S[:k], k, allowed)[1])
    res[var] = dict(Ks=Ks, curves=curves, S=S, order=order, P=P)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
titles = {"A": "а) прирост охвата сети МегаФона", "B": "б) выигранный охват (maximum capture)"}
for ax, var in zip(axes, ("A", "B")):
    Ks, c = res[var]["Ks"], res[var]["curves"]
    ax.fill_between(Ks, np.array(c["greedy"]) / 1e3, np.array(c["ub"]) / 1e3, color=BLUE, alpha=0.08, linewidth=0)
    for key, col, lab in [("greedy", BLUE, "жадный (CELF)"), ("density", ORANGE, "по плотности"),
                          ("kmeans", AQUA, "взвешенный k-means"), ("random", GRAY, "случайный")]:
        y = np.array(c[key]) / 1e3
        ax.plot(Ks, y, color=col, linewidth=2, marker="o", markersize=3, label=lab)
        dy = {"A": {"kmeans": 5, "density": -5}, "B": {"greedy": 6, "density": -6, "kmeans": 6, "random": -6}}[var].get(key, 0)
        ax.annotate(lab, (Ks[-1], y[-1]), xytext=(4, dy), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.plot(Ks, np.array(c["ub"]) / 1e3, color=BLUE, linewidth=1, linestyle="--", label="верхняя оценка оптимума")
    ax.set_title(titles[var], fontsize=9, color=INK, loc="left")
    ax.set_xlabel("число новых станций K"); ax.set_ylabel("тыс. жителей")
    ax.set_xlim(0, 260)
axes[0].legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig("out/fig3_gain_vs_k.png"); plt.close(fig)

P = res["A"]["P"]
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
unc = (P.v0 == 0) & (cells["pop"].values > 0)
for ax, (sel, col, title) in zip(axes, [(res["A"]["S"][:50], BLUE, "а) жадный алгоритм, 50 станций"),
                                        (res["A"]["order"][:50], ORANGE, "б) по плотности, 50 станций")]):
    basemap(ax)
    ax.scatter(cells.x[unc], cells.y[unc], s=1.2, color="#9ec5f4", linewidths=0, rasterized=True)
    ax.scatter(cand.x.values[sel], cand.y.values[sel], s=28, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_title(f"{title}: +{P.value(sel)/1e3:.1f} тыс. жителей", fontsize=9, color=INK)
axes[0].scatter([], [], s=12, color="#9ec5f4", label="жители вне покрытия МегаФона")
axes[0].legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("out/fig2_megafon_sites.png"); plt.close(fig)

# ---------- Рис. 4: «с нуля» ----------
P, allowed, own = build(cells, cand, pairs, "MTS", "T", 135.0)
S, vals, _ = lazy_greedy(P, K=5300, allowed=allowed)
T = cells["pop"].sum()
fig, ax = plt.subplots(figsize=(6.5, 3.8))
k = np.arange(1, len(vals) + 1)
ax.plot(k, np.array(vals) / T * 100, color=BLUE, linewidth=2, label="жадный выбор мест")
sc = pd.read_csv("out/scratch_th135_f1800.csv")
for _, r in sc.iterrows():
    ax.plot(r.N, r.real * 100, "o", color=ORANGE, markersize=6, markeredgecolor="white")
    ax.annotate("", (r.k_match, r.real * 100), (r.N, r.real * 100), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
ax.plot([], [], "o", color=ORANGE, label="реальные сети операторов")
txt = "\n".join(f"{OPNAME[r.op]}: {int(r.N)} ст. → {r.real*100:.1f}%; жадному хватает {int(r.k_match)}" for _, r in sc.iterrows())
ax.text(0.45, 0.62, txt, transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
ax.set_xscale("log"); ax.set_ylim(60, 100.5); ax.set_xlim(50, 12000)
ax.set_xlabel("число станций (лог. шкала)"); ax.set_ylabel("доля жителей в зоне покрытия, %")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig("out/fig4_scratch.png"); plt.close(fig)
print("готово")
