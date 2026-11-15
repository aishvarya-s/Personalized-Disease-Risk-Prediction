"""
LSTM / GRU for the 6-hour-ahead Sepsis-3 task, same patient-level 5-fold GroupKFold as baseline_sepsis3.py.
Run from project root:
    python src/models/train_lstm.py --model lstm --folds 1     # time one fold first
    python src/models/train_lstm.py --model gru  --folds 5
    python src/models/train_lstm.py --model both --folds 5

Windows (last 24 hours per prediction hour) are built lazily per batch from one flat float32 array,
so memory stays near the size of the dataset (about 1.2 GB) instead of 24x that.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["lstm", "gru", "both"], default="both")
ap.add_argument("--folds", type=int, default=5, help="number of folds to RUN (of 5, in order)")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--out", default="results")
args = ap.parse_args()

SEQ_LEN, NEG_RATIO, POS_WEIGHT, SEED = 24, 20, 20.0, 42
VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "mean_bp", "respiratory_rate", "spo2", "temperature"]
LABS = ["lactate", "ph", "wbc", "platelet", "hemoglobin", "creatinine", "bun", "sodium", "potassium",
        "bicarbonate", "glucose", "chloride", "bilirubin_total", "inr", "ptt"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "(NO GPU - this will be slow)")

# ---------------- data ----------------
df = pd.read_parquet("data/processed/sepsis3_dataset.parquet")
# rows must be contiguous and ordered by hour within each stay for the window lookup to work
df = df.sort_values(["stay_id", "hour_idx"], kind="stable").reset_index(drop=True)
feats = (VITALS + ["shock_index", "pulse_pressure", "gender", "anchor_age", "hour_idx"]
         + LABS + [f"{l}_hrs_since" for l in LABS])
missing = [c for c in feats if c not in df.columns]
assert not missing, f"columns not found in parquet: {missing}"
X = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
y = df["label"].to_numpy(np.int8)
groups = df["subject_id"].to_numpy()
stay = df["stay_id"].to_numpy()
hour = df["hour_idx"].to_numpy()
n, F = X.shape
# row i's window = rows [i-23 .. i] of the same stay. A row has `avail` earlier rows in its stay
# (only valid if hours are consecutive, which the preprocessing grid guarantees; checked below).
first = np.r_[True, stay[1:] != stay[:-1]]
start_idx = np.maximum.accumulate(np.where(first, np.arange(n), 0))
avail = (np.arange(n) - start_idx).astype(np.int64)           # rows available before this one in the stay
# sanity: hours are consecutive within a stay, but rows after onset are dropped, so only a tail is missing
assert (hour[~first] - hour[np.where(~first)[0] - 1] == 1).all(), "gaps in hour_idx inside a stay"
print(f"rows {n} | features {F}+1 pad flag | positives {int(y.sum())} ({y.mean():.4%})")
del df

OFF = np.arange(-SEQ_LEN + 1, 1)                               # -23..0


def make_batch(idx, mean, std):
    """Build (B, 24, F+1) windows for row indices idx, normalised with TRAIN statistics only."""
    pos = idx[:, None] + OFF[None, :]                          # (B, 24) absolute row positions
    pad = OFF[None, :] < -avail[idx][:, None]                  # True where before the stay started
    pos = np.where(pad, idx[:, None], pos)                     # point padded steps at a valid row, then mask
    w = X[pos]                                                 # (B, 24, F)
    miss = np.isnan(w)
    w = np.where(miss, mean, w)
    w = (w - mean) / std
    w[pad] = 0.0
    out = np.concatenate([w, pad[..., None].astype(np.float32)], axis=2)
    return torch.from_numpy(out.astype(np.float32))


class Net(nn.Module):
    def __init__(self, kind, in_dim, hidden):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)                # last time step


@torch.no_grad()
def predict(model, idx, mean, std):
    model.eval()
    out = []
    for i in range(0, len(idx), 4096):
        xb = make_batch(idx[i:i + 4096], mean, std).to(dev, non_blocking=True)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)


def metrics(yt, p):
    yt = yt == 1
    r = {"auroc": roc_auc_score(yt, p), "auprc": average_precision_score(yt, p)}
    for rate in (0.005, 0.01, 0.02, 0.05, 0.10):
        pred = p >= np.quantile(p, 1 - rate)
        tp = (pred & yt).sum()
        r[f"recall@{rate:.1%}"] = tp / max(yt.sum(), 1)
        r[f"precision@{rate:.1%}"] = tp / max(pred.sum(), 1)
    return r


def run_fold(kind, k, tr, te):
    rng = np.random.default_rng(SEED + k)
    torch.manual_seed(SEED + k)
    # 10% of training PATIENTS for early stopping (never the test fold)
    tr_pat = np.unique(groups[tr])
    val_pat = set(rng.choice(tr_pat, size=max(1, len(tr_pat) // 10), replace=False))
    is_val = np.fromiter((g in val_pat for g in groups[tr]), bool, count=len(tr))
    fit_rows, val_rows = tr[~is_val], tr[is_val]
    # normalisation statistics from the training rows only
    mean = np.nanmean(X[fit_rows], 0).astype(np.float32)
    mean = np.nan_to_num(mean)
    std = np.nanstd(X[fit_rows], 0).astype(np.float32)
    std = np.where((std < 1e-6) | np.isnan(std), 1.0, std)
    mean_f, std_f = mean, std            # shape (F,); the pad flag is appended after normalisation

    model = Net(kind, F + 1, args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=dev))
    pos_i, neg_i = fit_rows[y[fit_rows] == 1], fit_rows[y[fit_rows] == 0]
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        ep_rows = np.concatenate([pos_i, rng.choice(neg_i, size=min(len(neg_i), NEG_RATIO * len(pos_i)), replace=False)])
        rng.shuffle(ep_rows)
        tot = 0.0
        for i in range(0, len(ep_rows), args.batch):
            b = ep_rows[i:i + args.batch]
            xb = make_batch(b, mean_f, std_f).to(dev, non_blocking=True)
            yb = torch.from_numpy(y[b].astype(np.float32)).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        # validation AUROC on a fixed random subsample of validation-patient rows (all positives + 20x negatives)
        vp = val_rows[y[val_rows] == 1]
        vn = val_rows[y[val_rows] == 0]
        vrng = np.random.default_rng(SEED)
        vidx = np.concatenate([vp, vrng.choice(vn, size=min(len(vn), NEG_RATIO * max(len(vp), 1)), replace=False)])
        va = roc_auc_score(y[vidx], predict(model, vidx, mean_f, std_f))
        print(f"  [{kind} fold {k+1}] epoch {ep+1:2d} loss {tot/len(ep_rows):.4f} val AUROC {va:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best + 1e-4:
            best, bad = va, 0
            best_state = {a: b.detach().clone() for a, b in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                print("  early stop"); break
    model.load_state_dict(best_state)
    p = predict(model, te, mean_f, std_f)
    m = metrics(y[te], p)
    m["best_val_auroc"] = best
    return m, p


Path(args.out).mkdir(parents=True, exist_ok=True)
kinds = ["lstm", "gru"] if args.model == "both" else [args.model]
splits = list(GroupKFold(n_splits=5).split(X, y, groups))        # same call as the baseline -> identical folds
for kind in kinds:
    allm = []
    for k in range(min(args.folds, 5)):
        tr, te = splits[k]
        t0 = time.time()
        m, p_te = run_fold(kind, k, tr, te)
        np.savez_compressed(Path(args.out) / f"{kind}_fold{k+1}.npz", stay=stay[te], hour=hour[te], y=y[te], p=p_te)
        m["fold"], m["minutes"] = k + 1, (time.time() - t0) / 60
        allm.append(m)
        json.dump(allm, open(Path(args.out) / f"{kind}_folds.json", "w"), indent=1, default=float)   # resume-friendly
        print(f"{kind} fold {k+1}: AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.4f} recall@5% {m['recall@5.0%']:.3f} ({m['minutes']:.1f} min)", flush=True)
    a = np.array([[m["auroc"], m["auprc"], m["recall@5.0%"]] for m in allm])
    print(f"\n{kind.upper()} mean over {len(allm)} fold(s): AUROC {a[:,0].mean():.3f} +- {a[:,0].std():.3f} | "
          f"AUPRC {a[:,1].mean():.4f} | recall@5% {a[:,2].mean():.3f}\n")