import numpy as np
import pandas as pd
from opt import Problem

OPS = ["MTS", "Beeline", "MegaFon", "T2"]
DELTA = 1.0  # дБ: разница сигналов, которая считается «ничьей»
COST_COLOC, COST_NEW = 1.0, 2.5


def load(tag="f1800", demand=None):
    cells = pd.read_parquet("out/cells.parquet")
    cand = pd.read_parquet("out/cand.parquet")
    pairs = pd.read_parquet(f"out/pairs_{tag}.parquet")
    if demand is not None:  # пересчёт весов ячеек по другой модели спроса
        cells["pop"] = cells.h.map(demand).fillna(0).values
    return cells, cand, pairs


def _val(pl, plc, th):
    """1 — сигнал лучше всех конкурентов, 1/2 — ничья, 0 — хуже или нет покрытия."""
    v = np.where(pl < plc - DELTA, 1.0, np.where(pl <= plc + DELTA, 0.5, 0.0))
    return np.where(pl <= th, v, 0.0)


def build(cells, cand, pairs, op, variant="A", th=135.0, cand_mask=None, exclude_sites=None):
    """variant A — прирост охвата оператора op; B — выигранный охват (maximum capture);
    variant T — полный охват «с нуля» (v0 = 0, все кандидаты)."""
    nb = len(cells)
    p = pairs[pairs.pl <= th]
    has_op = cand.ops.str.contains(op).values
    if exclude_sites is not None:
        has_op = has_op & ~np.isin(np.arange(len(cand)), exclude_sites)
    is_exist = cand.kind.eq("existing").values
    own = has_op & is_exist
    comp = is_exist & (cand.ops.str.replace(op, "").str.strip(",").str.len() > 0).values
    comp_mask = np.zeros(len(cand), bool); comp_mask[:] = comp
    # для колокации у «чужой» площадки конкуренты — это все операторы на ней, кроме op
    plX = np.full(nb, np.inf); plC = np.full(nb, np.inf)
    po = p[own[p.c.values]]
    np.minimum.at(plX, po.b.values, po.pl.values)
    # конкуренты: площадки, где есть хоть один оператор кроме op
    pc = p[comp_mask[p.c.values]]
    np.minimum.at(plC, pc.b.values, pc.pl.values)

    if variant == "T":
        v0 = np.zeros(nb)
        allowed_mask = np.ones(len(cand), bool)
    else:
        allowed_mask = ~own
        if variant == "A":
            v0 = (plX <= th).astype(float)
        else:
            v0 = _val(plX, plC, th)
    if cand_mask is not None:
        allowed_mask &= cand_mask
    allowed = np.where(allowed_mask)[0]

    p2 = p[allowed_mask[p.c.values]].sort_values("c")
    if variant in ("A", "T"):
        vals = np.ones(len(p2))
    else:
        vals = _val(p2.pl.values, plC[p2.b.values], th)
    keep = vals > 0
    p2 = p2[keep]; vals = vals[keep]
    starts = np.searchsorted(p2.c.values, np.arange(len(cand) + 1))
    cc = [p2.b.values[starts[j]:starts[j + 1]] for j in range(len(cand))]
    cv = [vals[starts[j]:starts[j + 1]] for j in range(len(cand))]
    cost = np.where(is_exist, COST_COLOC, COST_NEW)
    P = Problem(cells["pop"].values, v0, cc, cv, cost)
    return P, allowed, own
