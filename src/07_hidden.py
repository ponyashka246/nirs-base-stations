"""Тест «скрытая часть сети»: прячем 15% площадок оператора в ЛО и смотрим, сколько из них алгоритм восстанавливает
(выбранное место ближе 1 км к спрятанной площадке)."""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from common import load, build, OPS
from opt import lazy_greedy

cells, cand, pairs = load("f1800")
lo_c = cand.lat.values > 0  # заглушка, ниже — по принадлежности к ЛО через ближайшую ячейку
tree = BallTree(np.c_[cells.x.values, cells.y.values])
_, nn = tree.query(np.c_[cand.x.values, cand.y.values], k=1)
cand_lo = cells.region.values[nn[:, 0]] == "Ленинградская область"
rows = []
for op in OPS:
    own_all = np.where(cand.ops.str.contains(op).values & cand.kind.eq("existing").values & cand_lo)[0]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        hid = rng.choice(own_all, size=int(0.15 * len(own_all)), replace=False)
        P, allowed, own = build(cells, cand, pairs, op, "A", 135.0, exclude_sites=hid)
        allowed = allowed[cand_lo[allowed]]
        K = len(hid)
        S, _, _ = lazy_greedy(P, K=K, allowed=allowed)
        ht = BallTree(np.c_[cand.x.values[hid], cand.y.values[hid]])

        def hit(sel):
            d, _ = ht.query(np.c_[cand.x.values[sel], cand.y.values[sel]], k=1)
            return float((d[:, 0] < 1000).mean())
        rnd = np.mean([hit(rng.choice(allowed, K, replace=False)) for _ in range(10)])
        stand = P.standalone()
        dens = hit(allowed[np.argsort(-stand[allowed])[:K]])
        rows.append(dict(op=op, seed=seed, K=K, greedy=hit(S), density=dens, random=rnd, lost=P.value(hid), restored_greedy=P.value(S), restored_density=P.value(allowed[np.argsort(-stand[allowed])[:K]])))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r.to_csv("out/hidden.csv", index=False)
print(r.groupby("op")[["greedy", "density", "random", "lost", "restored_greedy", "restored_density"]].mean().round(3))
print(r[["greedy", "density", "random", "lost", "restored_greedy", "restored_density"]].mean().round(3))
