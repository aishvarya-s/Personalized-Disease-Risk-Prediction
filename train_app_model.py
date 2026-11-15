"""
Model for the web app (predict.py): snapshot vitals + age/sex -> risk of Sepsis-3 onset within the next 6 hours.
Replaces train_and_save.py (which used the old admission-level ICD label).

Patients are split 60/20/20 into train / calibration / test:
  train        -> fit XGBoost (all positives + 20x negatives)
  calibration  -> (1) Platt-scale scores into real probabilities, (2) choose the High / Moderate cut-offs
  test         -> report performance once, with those cut-offs
Run from project root:  python train_app_model.py
"""
import json, os, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

FEATURES = ["heart_rate", "systolic_bp", "diastolic_bp", "mean_bp", "shock_index", "pulse_pressure",
            "respiratory_rate", "spo2", "gender", "anchor_age"]          # same inputs the app form collects
HIGH_Q, MOD_Q = 0.95, 0.80           # High = top 5% of scores, Moderate = next 15%  (chosen on calibration patients)

df = pd.read_parquet("data/processed/sepsis3_dataset.parquet", columns=["subject_id", "label"] + FEATURES)
X, y, g = df[FEATURES].replace([np.inf, -np.inf], np.nan), df["label"].values, df["subject_id"].values

rest, te = next(GroupShuffleSplit(1, test_size=0.20, random_state=42).split(X, y, g))
a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(rest, groups=g[rest]))
tr, cal = rest[a], rest[b]
print("patients  train %d | calibration %d | test %d" % tuple(len(set(g[i])) for i in (tr, cal, te)))
print("positives train %d | calibration %d | test %d" % tuple(int(y[i].sum()) for i in (tr, cal, te)))

means = X.iloc[tr].mean()                                               # train statistics only
fill = lambda i: X.iloc[i].fillna(means)
scaler = StandardScaler().fit(fill(tr))

rng = np.random.default_rng(42)
pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
fit = np.concatenate([pos, rng.choice(neg, size=min(len(neg), 20 * len(pos)), replace=False)])
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=20, tree_method="hist", eval_metric="logloss", random_state=42)
model.fit(scaler.transform(fill(fit)), y[fit])

raw = lambda i: model.predict_proba(scaler.transform(fill(i)))[:, 1]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# Platt scaling on calibration patients (undersampling inflated the raw scores)
platt = LogisticRegression(C=1e6).fit(logit(raw(cal)).reshape(-1, 1), y[cal])
a_, b_ = float(platt.coef_[0, 0]), float(platt.intercept_[0])
calibrate = lambda r: 1 / (1 + np.exp(-(a_ * logit(r) + b_)))

p_cal, p_te = calibrate(raw(cal)), calibrate(raw(te))
high_thr, mod_thr = float(np.quantile(p_cal, HIGH_Q)), float(np.quantile(p_cal, MOD_Q))   # thresholds from calibration set

base = float(y[te].mean())
print("\nTEST (held-out patients): prevalence %.4f | AUROC %.3f | AUPRC %.4f" %
      (base, roc_auc_score(y[te], p_te), average_precision_score(y[te], p_te)))
print("mean calibrated probability %.4f vs observed rate %.4f" % (p_te.mean(), base))
for name, thr in [("High", high_thr), ("Moderate+", mod_thr)]:
    f = p_te >= thr
    print("%-9s threshold %.4f -> flags %.1f%% of rows | recall %.3f | precision %.4f (%.1fx base rate)" %
          (name, thr, 100 * f.mean(), y[te][f].sum() / y[te].sum(), y[te][f].mean(), y[te][f].mean() / base))

os.makedirs("src/models/saved", exist_ok=True)
for fn, obj in [("xgboost_model.pkl", model), ("scaler.pkl", scaler), ("train_means.pkl", means)]:
    pickle.dump(obj, open(f"src/models/saved/{fn}", "wb"))
json.dump(dict(platt_a=a_, platt_b=b_, high_thr=high_thr, moderate_thr=mod_thr, base_rate=float(y[cal].mean()),
               test_auroc=float(roc_auc_score(y[te], p_te)), test_auprc=float(average_precision_score(y[te], p_te))),
          open("src/models/saved/thresholds.json", "w"), indent=2)
print("\nSaved model, scaler, train_means and thresholds.json to src/models/saved/")