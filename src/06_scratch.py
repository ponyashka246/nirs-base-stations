"""Протокол «с нуля»: столько же станций, сколько у оператора, но места выбирает алгоритм
(из всех существующих площадок и новых кандидатов). Плюс: сколько станций нужно, чтобы повторить охват реальной сети."""
import sys
import numpy as np
import pandas as pd
from common import load, build, OPS
from opt import lazy_greedy

TH = float(sys.argv[1]) if len(sys.argv) > 1 else 135.0
TAG = sys.argv[2] if len(sys.argv) > 2 else "f1800"
cells, cand, pairs = load(TAG)
T = cells["pop"].sum()
lo = cells.region.eq("Ленинградская область").values
rows = []
for op in OPS:
    P, allowed, own = build(cells, cand, pairs, op, "T", TH)
    real = P.value(np.where(own)[0])
    N = int(own.sum())
    S, vals, cur = lazy_greedy(P, K=N, allowed=allowed)
    vals = np.array(vals)
    k_match = int(np.searchsorted(vals, real) + 1) if vals[-1] >= real else np.nan
    cur_real = P.v0.copy()
    for j in np.where(own)[0]:
        P.add(j, cur_real)
    lo_real = (P.w * cur_real)[lo].sum() / P.w[lo].sum()
    lo_opt = (P.w * cur)[lo].sum() / P.w[lo].sum()
    rows.append(dict(op=op, N=N, real=real / T, greedy_N=vals[-1] / T, k_match=k_match, k_match_share=k_match / N,
                     lo_real=lo_real, lo_greedy=lo_opt, share_new=float((cand.kind.values[S] == "new").mean())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"out/scratch_th{int(TH)}_{TAG}.csv", index=False)
