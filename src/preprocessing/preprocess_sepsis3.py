"""
Final time-aligned Sepsis-3 dataset: "will this ICU patient develop sepsis within the next 6 hours?"

Inputs (data/raw/):
    vitals_all.csv  + any vitals_part_*.csv      (hourly vitals, first 72 ICU hours)
    labs_all.csv    + any labs_part_*.csv        (lab draws, from 24h before to 72h after ICU admission)
    sepsis3_all.csv, icustays.csv, patients.csv
Output: data/processed/sepsis3_dataset.parquet   (one row per ICU stay-hour, float32)

Run from project root:  python src/preprocessing/preprocess_sepsis3.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)      # all-NaN windows are expected
ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"

HORIZON_H = 6            # predict onset within the next 6 hours
MIN_ONSET_H = 6          # stays with onset in the first 6 ICU hours are excluded (no pre-onset window)
VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "mean_bp", "respiratory_rate", "spo2", "temperature"]
RANGES = {"heart_rate": (20, 250), "systolic_bp": (30, 300), "diastolic_bp": (10, 200), "mean_bp": (20, 250),
          "respiratory_rate": (4, 60), "spo2": (50, 100), "temperature": (30, 43)}
LAB_RANGES = {"lactate": (0.2, 30), "ph": (6.8, 7.8), "wbc": (0, 500), "platelet": (0, 2000), "hemoglobin": (2, 25),
              "creatinine": (0.1, 30), "bun": (1, 300), "sodium": (100, 180), "potassium": (1.5, 10),
              "bicarbonate": (2, 60), "glucose": (10, 1500), "chloride": (60, 150), "bilirubin_total": (0.1, 80),
              "inr": (0.5, 20), "ptt": (15, 300)}
LABS = list(LAB_RANGES)
LAB_DELAY_H = 1          # a lab drawn at charttime is only known ~1h later -> never use it earlier (no leakage)
NEVER = 99               # "hours since last measured" for a lab not yet measured
WIN_VITALS = ["heart_rate", "systolic_bp", "mean_bp", "respiratory_rate", "spo2", "temperature"]
TREND_LABS = ["lactate", "creatinine", "wbc", "platelet", "bilirubin_total", "ph", "bicarbonate"]
COUNT_VITALS = ["heart_rate", "systolic_bp", "respiratory_rate", "spo2", "temperature"]


def to_dt(s):
    """Parse timestamps with or without 'T' / '+00:00'; return naive datetimes."""
    return pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_localize(None)


def read_parts(stem, time_cols, **kw):
    files = [f for f in [RAW / f"{stem}_all.csv"] + sorted(RAW.glob(f"{stem}_part_*.csv")) if f.exists()]
    print("  %s files: %s" % (stem, [f.name for f in files]))
    out = []
    for f in files:
        d = pd.read_csv(f, **kw)
        for c in time_cols:
            d[c] = to_dt(d[c])
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------- vectorised window helpers (rows are sorted by stay, then hour; hh = hour index in stay) ----------------
def windows(a, hh, w):
    """(N,w) matrix: the last w values ending at each row; NaN where the window would cross into the previous stay."""
    pad = np.concatenate([np.full(w - 1, np.nan, np.float32), a])
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    ok = hh[:, None] >= (w - 1 - np.arange(w))[None, :]
    return np.where(ok, win, np.nan).astype(np.float32, copy=False)

def shift(a, hh, k):
    s = np.full_like(a, np.nan); s[k:] = a[:-k]; s[hh < k] = np.nan
    return s

def roll_count(flag, hh, w):
    cs0 = np.concatenate([[0.0], np.cumsum(flag, dtype=np.float64)])
    i = np.arange(len(flag)); m = np.minimum(hh + 1, w)
    return (cs0[i + 1] - cs0[i + 1 - m]).astype(np.float32)


# ---------------- load ----------------
print("Loading data...")
v = read_parts("vitals", ["hour"], dtype={c: "float32" for c in VITALS})
v = v.drop_duplicates(["stay_id", "hour"])
for c, (lo, hi) in RANGES.items():                       # remove charting errors
    v.loc[~v[c].between(lo, hi), c] = np.nan

labs = read_parts("labs", ["charttime"], dtype={c: "float32" for c in LABS}).drop_duplicates()
for c, (lo, hi) in LAB_RANGES.items():
    labs.loc[~labs[c].between(lo, hi), c] = np.nan

stays = pd.read_csv(RAW / "icustays.csv", usecols=["stay_id", "intime"])
stays["intime"] = to_dt(stays["intime"])
stays["t0"] = stays["intime"].dt.floor("h")              # hour grid starts at the ICU admission hour

pat = pd.read_csv(RAW / "patients.csv", usecols=["subject_id", "gender", "anchor_age"])
pat["gender"] = (pat["gender"] == "M").astype(np.float32)

s3 = pd.read_csv(RAW / "sepsis3_all.csv", usecols=["stay_id", "suspected_infection_time", "sepsis3"])
s3 = s3[s3["sepsis3"].astype(str).str.lower() == "true"].copy()
s3["onset"] = to_dt(s3["suspected_infection_time"])
s3 = s3.sort_values("onset").drop_duplicates("stay_id")[["stay_id", "onset"]]

# ---------------- cohort ----------------
v = v.merge(stays[["stay_id", "t0"]], on="stay_id", how="inner")
v = v.merge(pat, on="subject_id", how="left")
v = v[v["anchor_age"] >= 18]                              # adults only
v["h"] = ((v["hour"] - v["t0"]).dt.total_seconds() // 3600).astype(int)
v = v[(v["h"] >= 0) & (v["h"] <= 72)]

s3 = s3.merge(stays, on="stay_id")
s3["onset_h"] = (s3["onset"] - s3["t0"]).dt.total_seconds() / 3600
s3["onset_since_icu_h"] = (s3["onset"] - s3["intime"]).dt.total_seconds() / 3600
too_early = s3.loc[s3["onset_since_icu_h"] < MIN_ONSET_H, "stay_id"]
print("Sepsis-3 stays excluded (onset < %dh after ICU admission): %d" % (MIN_ONSET_H, len(too_early)))
v = v[~v["stay_id"].isin(too_early)]
onset_h = s3.set_index("stay_id")["onset_h"]
static = v.drop_duplicates("stay_id").set_index("stay_id")[["subject_id", "gender", "anchor_age"]]

# ---------------- hourly grid per stay (0 .. last charted hour) ----------------
print("Building hourly grid...")
last_h = v.groupby("stay_id")["h"].max().sort_index()
cnt = last_h.values + 1
grid = pd.DataFrame({"stay_id": np.repeat(last_h.index.values, cnt)})
grid["h"] = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
v = grid.merge(v[["stay_id", "h"] + VITALS], on=["stay_id", "h"], how="left")
del grid
for c in ["subject_id", "gender", "anchor_age"]:
    v[c] = v["stay_id"].map(static[c])
hh = v["h"].values

# ---------------- labs: usable LAB_DELAY_H after the draw; pre-ICU labs known from hour 0 ----------------
print("Joining labs...")
labs = labs.merge(stays[["stay_id", "t0"]], on="stay_id")
labs["h"] = (((labs["charttime"] + pd.Timedelta(hours=LAB_DELAY_H)) - labs["t0"]).dt.total_seconds() // 3600).astype(int)
labs["h"] = labs["h"].clip(lower=0)
labs = labs.sort_values("charttime").groupby(["stay_id", "h"])[LABS].last().reset_index()   # last non-null per hour
v = v.merge(labs, on=["stay_id", "h"], how="left")
del labs

# charting / ordering behaviour flags (taken BEFORE forward-filling)
flags = {c: v[c].notna().values.astype(np.float32) for c in COUNT_VITALS}
lab_any = v[LABS].notna().any(axis=1).values.astype(np.float32)
for c in LABS:
    v[f"{c}_obs_h"] = v["h"].where(v[c].notna())

# forward-fill only (never back-fill: no future values); earlier gaps stay NaN
ff = VITALS + LABS + [f"{c}_obs_h" for c in LABS]
v[ff] = v.groupby("stay_id")[ff].ffill()
for c in LABS:
    v[f"{c}_hrs_since"] = (v["h"] - v[f"{c}_obs_h"]).fillna(NEVER)
    v = v.drop(columns=f"{c}_obs_h")

# ---------------- features ----------------
print("Building features...")
f32 = lambda x: np.asarray(x, dtype=np.float32)
feat = {"shock_index": f32(v["heart_rate"] / (v["systolic_bp"] + 1)),
        "pulse_pressure": f32(v["systolic_bp"] - v["diastolic_bp"])}
for c in WIN_VITALS:
    a = v[c].values.astype(np.float32)
    feat[f"{c}_mean6"] = np.nanmean(windows(a, hh, 6), 1)
    feat[f"{c}_delta6"] = a - shift(a, hh, 6)
    w12, w24 = windows(a, hh, 12), windows(a, hh, 24)
    feat[f"{c}_min12"], feat[f"{c}_max12"], feat[f"{c}_std12"] = np.nanmin(w12, 1), np.nanmax(w12, 1), np.nanstd(w12, 1)
    feat[f"{c}_min24"], feat[f"{c}_max24"], feat[f"{c}_std24"] = np.nanmin(w24, 1), np.nanmax(w24, 1), np.nanstd(w24, 1)
    feat[f"{c}_mean24"] = np.nanmean(w24, 1)
    del w12, w24
for c in TREND_LABS:                                       # change vs 12 hours ago
    a = v[c].values.astype(np.float32)
    feat[f"{c}_delta12"] = a - shift(a, hh, 12)
for c in COUNT_VITALS:                                     # how often is this vital being charted?
    feat[f"n_{c}_6h"], feat[f"n_{c}_24h"] = roll_count(flags[c], hh, 6), roll_count(flags[c], hh, 24)
feat["n_lab_hours_12h"], feat["n_lab_hours_24h"] = roll_count(lab_any, hh, 12), roll_count(lab_any, hh, 24)
v = pd.concat([v, pd.DataFrame({k: f32(x) for k, x in feat.items()}, index=v.index)], axis=1)

# ---------------- label ----------------
# A row at hour h summarises data up to the END of that hour, so the prediction time is t = h + 1.
# label = 1 if onset is within (t, t + 6];  rows with onset <= t (already septic) are dropped.
t = v["h"].values + 1
on = v["stay_id"].map(onset_h).values
v["label"] = ((on > t) & (on <= t + HORIZON_H)).astype(np.int8)
v = v[~(on <= t)].rename(columns={"h": "hour_idx"})

num = [c for c in v.columns if c not in ("stay_id", "subject_id", "label")]
v[num] = v[num].astype(np.float32)
OUT.mkdir(parents=True, exist_ok=True)
v.reset_index(drop=True).to_parquet(OUT / "sepsis3_dataset.parquet", index=False)

print("\nSaved:", OUT / "sepsis3_dataset.parquet")
print("Rows: %d | stays: %d | patients: %d | features: %d" %
      (len(v), v["stay_id"].nunique(), v["subject_id"].nunique(), len(num)))
print("Positive rows: %d (%.3f%%) | stays with >=1 positive row: %d | positive patients: %d" %
      (v["label"].sum(), 100 * v["label"].mean(), v.loc[v.label == 1, "stay_id"].nunique(),
       v.loc[v.label == 1, "subject_id"].nunique()))