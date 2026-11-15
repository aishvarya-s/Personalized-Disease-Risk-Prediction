"""
Ensemble of saved per-fold test predictions (same patient-level folds for every model).
Run from project root, after saving predictions from each model, e.g.
    python src/models/baseline_sepsis3.py --models XGBoost,RandForest --save-preds preds
    python src/models/train_lstm.py --model both --folds 5 --out preds
    python src/models/ensemble.py --dir preds --members XGBoost,gru
    python src/models/ensemble.py --dir preds --members XGBoost,gru,lstm,RandForest

Method: scores from different models are on different scales, so each model's scores are converted to
rank percentiles WITHIN the test fold (no labels used) and averaged with EQUAL weights. Weights are fixed
in advance on purpose: tuning them on the test folds would be a form of leakage.
"""
import argparse, itertools
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="preds")
ap.add_argument("--members", default="XGBoost,gru", help="comma list of model names (file prefixes)")
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()
members = args.members.split(",")
RATES = [0.005, 0.01, 0.02, 0.05, 0.10]


def load(name, k):
    d = np.load(f"{args.dir}/{name}_fold{k}.npz")
    key = d["stay"].astype(np.int64) * 1000 + d["hour"].astype(np.int64)    # hour < 1000
    o = np.argsort(key)
    return key[o], d["y"][o], d["p"][o].astype(np.float64)


def metrics(y, p):
    yt = y == 1
    r = [roc_auc_score(yt, p), average_precision_score(yt, p)]
    for rate in RATES:
        pred = p >= np.quantile(p, 1 - rate)
        tp = (pred & yt).sum()
        r += [tp / yt.sum(), tp / max(pred.sum(), 1)]
    return np.array(r)


names = ["AUROC", "AUPRC"] + [f"{t}@{r:.1%}" for r in RATES for t in ("rec", "prec")]
combos = {m: (m,) for m in members}
combos["ENSEMBLE(" + "+".join(members) + ")"] = tuple(members)
if len(members) > 2:                                   # also every pair, for reference (fixed, not tuned)
    for c in itertools.combinations(members, 2):
        combos["pair(" + "+".join(c) + ")"] = c
res = {c: [] for c in combos}
for k in range(1, args.folds + 1):
    data = {m: load(m, k) for m in members}
    key0, y0, _ = data[members[0]]
    for m in members[1:]:                              # same rows, same labels, or the folds don't match
        assert np.array_equal(key0, data[m][0]) and np.array_equal(y0, data[m][1]), \
            f"fold {k}: {m} rows/labels differ from {members[0]} (different folds or data?)"
    pct = {m: rankdata(data[m][2]) / len(y0) for m in members}
    for c, ms in combos.items():
        res[c].append(metrics(y0, np.mean([pct[m] for m in ms], 0)))

print("Mean over %d folds (rows = same test patients for every model)\n" % args.folds)
print("%-34s %-15s %-8s %s" % ("model", "AUROC (sd)", "AUPRC", "recall/precision @ 0.5% 1% 2% 5% 10%"))
for c, r in res.items():
    r = np.array(r)
    a = r.mean(0)
    rp = "  ".join("%.3f/%.3f" % (a[2 + 2 * i], a[3 + 2 * i]) for i in range(len(RATES)))
    print("%-34s %.3f (%.3f)   %.4f   %s" % (c, a[0], r[:, 0].std(), a[1], rp))
ens = [c for c in combos if c.startswith("ENSEMBLE")][0]
print("\nPer-fold AUROC and paired difference of the ensemble vs each member:")
E = np.array(res[ens])[:, 0]
for m in members:
    M = np.array(res[m])[:, 0]
    print("  vs %-10s members %s | ens %s | mean diff %+.4f (fold diffs %s)" % (
        m, np.round(M, 3), np.round(E, 3), (E - M).mean(), np.round(E - M, 4)))
print("\nIf the ensemble beats the best member in ALL folds, the gain is likely real; "
      "if it varies in sign, call it a tie.")