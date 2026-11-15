"""
Baselines for the 6-hour-ahead Sepsis-3 task, evaluated with patient-level 5-fold cross-validation.
Run from project root:
    python src/models/baseline_sepsis3.py             # all features
    python src/models/baseline_sepsis3.py --no-labs   # vitals only
    python src/models/baseline_sepsis3.py --no-timing # drop 'hours since' / 'how often charted' columns
    python src/models/baseline_sepsis3.py --basic     # original 54 features (no 12h/24h windows, lab trends, counts)
"""
import argparse, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
for f in ("--no-labs", "--no-timing", "--basic"):
    ap.add_argument(f, action="store_true")
ap.add_argument("--models", default="LogReg,RandForest,XGBoost", help="comma list to run, e.g. XGBoost")
ap.add_argument("--save-preds", default=None, help="folder: save each model's test-fold scores (for ensembling)")
args = ap.parse_args()

LABS = ["lactate", "ph", "wbc", "platelet", "hemoglobin", "creatinine", "bun", "sodium", "potassium",
        "bicarbonate", "glucose", "chloride", "bilirubin_total", "inr", "ptt"]
is_lab   = lambda c: c in LABS or any(c == f"{l}_hrs_since" or c == f"{l}_delta12" for l in LABS) or c.startswith("n_lab_")
is_timing = lambda c: c.endswith("_hrs_since") or c.startswith("n_")
is_new   = lambda c: re.search(r"_(min12|max12|std12|min24|max24|std24|mean24|delta12)$", c) or c.startswith("n_")

df = pd.read_parquet("data/processed/sepsis3_dataset.parquet")
feats = [c for c in df.columns if c not in ("stay_id", "subject_id", "label")]
mode = []
if args.no_labs:   feats = [c for c in feats if not is_lab(c)];    mode.append("no labs")
if args.no_timing: feats = [c for c in feats if not is_timing(c)]; mode.append("no timing/charting-frequency")
if args.basic:     feats = [c for c in feats if not is_new(c)];    mode.append("basic features")
ids = df[["stay_id", "hour_idx"]].to_numpy()
X, y, groups = df[feats].replace([np.inf, -np.inf], np.nan), df["label"].values, df["subject_id"].values
print("Mode: %s | rows %d | features %d | positive rate %.4f | positive patients %d" %
      (", ".join(mode) or "all features", len(df), len(feats), y.mean(), df.loc[df.label == 1, "subject_id"].nunique()))
del df

def models(pos_w):
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandForest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=50,
                                             class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=20, scale_pos_weight=pos_w,
                                 tree_method="hist", eval_metric="logloss", random_state=42),
    }

RATES = [0.01, 0.05, 0.10]                 # alert rates: flag the top 1% / 5% / 10% highest-risk hours
res, rec5, imp, ops = {}, {}, [], {}
rng = np.random.default_rng(42)
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):   # patients never span train and test
    # train on all positives + 20x as many random negatives (test fold stays complete, so evaluation is honest)
    pos_i, neg_i = tr[y[tr] == 1], tr[y[tr] == 0]
    fit = np.concatenate([pos_i, rng.choice(neg_i, size=min(len(neg_i), 20 * len(pos_i)), replace=False)])
    means = X.iloc[tr].mean().fillna(0)                                     # train statistics only
    Xfit, Xte = X.iloc[fit].fillna(means), X.iloc[te].fillna(means)
    sc = StandardScaler().fit(Xfit); Xfit, Xte = sc.transform(Xfit), sc.transform(Xte)
    yfit = y[fit]
    pos_w = (yfit == 0).sum() / max((yfit == 1).sum(), 1)
    for name, m in models(pos_w).items():
        if name not in args.models.split(","):
            continue
        m.fit(Xfit, yfit)
        p = m.predict_proba(Xte)[:, 1]
        if args.save_preds:
            import os; os.makedirs(args.save_preds, exist_ok=True)
            np.savez_compressed(f"{args.save_preds}/{name}_fold{k+1}.npz", stay=ids[te, 0], hour=ids[te, 1], y=y[te], p=p)
        res.setdefault(name, []).append((roc_auc_score(y[te], p), average_precision_score(y[te], p)))
        thr = np.quantile(p, 0.95)                                           # flag the top 5% highest-risk rows
        rec5.setdefault(name, []).append((p[y[te] == 1] >= thr).mean())
        yt = y[te] == 1
        for rate in RATES:                                                   # precision / recall / accuracy at fixed alert rates
            pred = p >= np.quantile(p, 1 - rate)
            tp, fp = (pred & yt).sum(), (pred & ~yt).sum()
            fn, tn = (~pred & yt).sum(), (~pred & ~yt).sum()
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            ops.setdefault((name, rate), []).append(
                (prec, rec, (tp + tn) / len(p), tn / max(tn + fp, 1), 2 * prec * rec / max(prec + rec, 1e-12)))
        if name == "XGBoost":
            imp.append(m.feature_importances_)
    print("fold", k + 1, "done")

print("\nprevalence (random-guess AUPRC): %.4f | random recall at 5%% alert rate: 0.050" % y.mean())
for n, r in res.items():
    r = np.array(r)
    print("%-11s AUROC %.3f +- %.3f | AUPRC %.4f +- %.4f | recall@5%% alerts %.3f" %
          (n, r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std(), np.mean(rec5[n])))
print("\nPRECISION / RECALL / ACCURACY on the held-out patients (mean over 5 folds).")
print("Threshold = flag the top X%% of hours by risk score. Always predicting 'no sepsis' gives accuracy %.4f." % (1 - y.mean()))
print("%-11s %-7s %-10s %-8s %-9s %-11s %-6s" % ("Model", "Alerts", "Precision", "Recall", "Accuracy", "Specificity", "F1"))
for n in res:
    for rate in RATES:
        a = np.mean(ops[(n, rate)], 0)
        print("%-11s %-7s %-10.4f %-8.3f %-9.4f %-11.3f %-6.4f" % (n, "%d%%" % round(rate * 100), *a))

if imp:
  top = pd.Series(np.mean(imp, 0), index=feats).sort_values(ascending=False).head(15)
  print("\nTop XGBoost features (average over folds):\n" + top.round(4).to_string())