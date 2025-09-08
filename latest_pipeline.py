# ============================================================
# CKD Thesis Pipeline — Bias-Aware (IFCM, C=3) + Hybrid MLP + Transformer
# with EDA after combining real + synthetic training data
# ============================================================
# Usage:
#   Train and evaluate:
#       python rithana_full_pipeline.py
#   Batch-score new patients:
#       python rithana_full_pipeline.py --score patients.csv
# ============================================================

import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, brier_score_loss
)
import joblib
from scipy import sparse as sp

ROOT   = Path.cwd()
PLOTS1 = ROOT / "plots1";    PLOTS1.mkdir(exist_ok=True)     # <— NEW plots folder
ART    = ROOT / "artifacts"; ART.mkdir(exist_ok=True)
RNG    = np.random.default_rng(42)

# --------------------------- utils ---------------------------

def savefig(name: str):
    """Save to plots1/name.png"""
    plt.tight_layout()
    plt.savefig(PLOTS1 / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()

def to_dense(X):
    if hasattr(X, "toarray"): return X.toarray()
    return np.asarray(X, dtype=np.float32)

def hstack_feature(Xm, col):
    col = np.asarray(col).reshape(-1, 1)
    if sp.issparse(Xm): return sp.hstack([Xm, col], format="csr")
    return np.hstack([Xm, col])

def write_md(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f: f.write(text)

# ------------------------ data cleaning ----------------------

def load_and_clean(csv_path="kidney_disease.csv"):
    df = pd.read_csv(csv_path)
    # normalize strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = (df[c].astype(str).str.strip().str.lower()
                 .replace({"?": np.nan, "nan": np.nan}))
    # labels
    df["classification"] = (df["classification"]
        .replace({"ckd\t":"ckd","ckd ":"ckd","notckd ":"notckd"})
        .map({"notckd":0,"ckd":1}).astype(int))
    if "id" in df.columns: df = df.drop(columns=["id"])
    return df

def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])
    return pre, num_cols, cat_cols

# --------------------- train-only augmentation ---------------------

def synth_bootstrap_train(X_tr: pd.DataFrame, y_tr: pd.Series,
                          num_cols, cat_cols, size_multiplier=0.8):
    """Bootstrap synthetic rows *from the training split only*."""
    df_g = X_tr.copy()
    if num_cols:
        df_g[num_cols] = SimpleImputer(strategy="median").fit_transform(df_g[num_cols])
    if cat_cols:
        df_g[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_g[cat_cols])
        df_g[cat_cols] = df_g[cat_cols].astype(str)

    base = pd.concat([df_g.reset_index(drop=True),
                      y_tr.reset_index(drop=True).rename("classification")], axis=1)
    n_syn = max(1, int(len(base) * size_multiplier))
    idx = RNG.integers(0, len(base), size=n_syn)
    synthetic = base.iloc[idx].copy()
    return synthetic

# --------------------- EDA after combining -------------------

def eda_after_combine(train_combined: pd.DataFrame):
    """
    EDA on combined (real + synthetic) training table.
    Expects columns: classification, source
    """
    df = train_combined.copy()
    df["source"] = df["source"].astype("category")

    # 1) Class balance overall + by source
    plt.figure(figsize=(4.8,3.6))
    sns.countplot(x="classification", data=df, palette="Blues")
    plt.xticks([0,1], ["notckd","ckd"]); plt.title("Class balance (combined)")
    savefig("01_eda_class_balance_combined")

    plt.figure(figsize=(6.4,3.6))
    sns.countplot(x="source", hue="classification", data=df, palette="Blues")
    plt.legend(title="class", labels=["notckd","ckd"])
    plt.title("Class balance by source (real vs synthetic)")
    savefig("02_eda_class_balance_by_source")

    # 2) KDE overlays (real vs synthetic) for numeric CKD features (only those present)
    #    Curated list; we’ll only plot if a column exists
    cand = ["age","bgr","hemo","pcv","sg","sc","bp"]
    num_cols = [c for c in cand if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        plt.figure(figsize=(5.2,3.8))
        try:
            sns.kdeplot(data=df, x=col, hue="source", fill=True, common_norm=False, alpha=0.4)
        except Exception:
            # fallback to hist if KDE fails
            for src in ["real","synthetic"]:
                subset = df[df["source"]==src][col].dropna()
                plt.hist(subset, bins=30, alpha=0.5, label=src)
            plt.legend()
        plt.title(f"Real vs Synthetic — {col}")
        savefig(f"10_eda_kde_{col}")

    # 3) Correlation heatmap on numeric columns
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] >= 2:
        plt.figure(figsize=(6,5))
        sns.heatmap(nums.corr(numeric_only=True), cmap="vlag", center=0, annot=False)
        plt.title("Numeric correlation heatmap (combined)")
        savefig("20_eda_corr_heatmap")

# --------------------- incremental FCM (C=3) ----------------------

class IncrementalFCM:
    """Lightweight incremental FCM with KMeans warm start."""
    def __init__(self, c=3, m=2.0, alpha=0.2, random_state=42):
        self.c=c; self.m=m; self.alpha=alpha
        self.rng=np.random.default_rng(random_state)
        self.centers_=None

    @staticmethod
    def _euclid2(a, b):
        a2=(a*a).sum(1,keepdims=True); b2=(b*b).sum(1,keepdims=True).T
        return np.maximum(a2 + b2 - 2*a@b.T, 1e-12)

    def warm_start(self, X):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.c, n_init="auto", random_state=42).fit(X)
        self.centers_ = km.cluster_centers_

    def partial_fit(self, X):
        D=self._euclid2(X,self.centers_)
        inv=(1.0/D)**(1.0/(self.m-1.0))
        U=inv/inv.sum(axis=1,keepdims=True)
        Um=U**self.m
        num=Um.T@X
        den=Um.sum(axis=0)[:,None]
        new=num/den
        self.centers_=(1-self.alpha)*self.centers_+self.alpha*new
        return U

    def predict_membership(self, X):
        D=self._euclid2(X,self.centers_)
        inv=(1.0/D)**(1.0/(self.m-1.0))
        return inv/inv.sum(axis=1,keepdims=True)

def run_ifcm_and_append(X_train_enc, X_test_enc):
    # learn centers on TRAIN
    ifcm = IncrementalFCM(c=3)                          # <— C=3
    ifcm.warm_start(to_dense(X_train_enc))
    ifcm.partial_fit(to_dense(X_train_enc))

    # membership → hard id
    U_tr = ifcm.predict_membership(to_dense(X_train_enc))
    U_te = ifcm.predict_membership(to_dense(X_test_enc))
    cl_tr = U_tr.argmax(1)
    cl_te = U_te.argmax(1)

    # save centers
    np.save(ART/"ifcm_centers.npy", ifcm.centers_)

    # append id
    Xh_train = hstack_feature(X_train_enc, cl_tr)
    Xh_test  = hstack_feature(X_test_enc,  cl_te)

    # composition plot (train clusters)
    plt.figure(figsize=(4.6,3.4))
    sns.countplot(x=cl_tr, palette="Blues")
    plt.title("IFCM cluster composition (train)"); plt.xlabel("cluster_id")
    savefig("30_ifcm_cluster_composition_train")

    return Xh_train, Xh_test, cl_tr, cl_te

# ----------------------- evaluation helpers ------------------------

def eval_and_plot(name, y_true, proba, pred):
    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["notckd","ckd"], yticklabels=["notckd","ckd"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(f"{name} — Confusion Matrix")
    savefig(f"{name.lower()}_confusion_matrix")

    RocCurveDisplay.from_predictions(y_true, proba)
    plt.title(f"{name} — ROC"); savefig(f"{name.lower()}_roc_curve"); plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, proba)
    plt.title(f"{name} — PR Curve"); savefig(f"{name.lower()}_pr_curve"); plt.close()

    try:
        from sklearn.calibration import CalibrationDisplay
        CalibrationDisplay.from_predictions(y_true, proba, n_bins=10)
        bri = brier_score_loss(y_true, proba)
        plt.title(f"{name} — Calibration (Brier={bri:.3f})")
        savefig(f"{name.lower()}_calibration"); plt.close()
    except Exception:
        bri = brier_score_loss(y_true, proba)

    return dict(
        acc = float(accuracy_score(y_true, pred)),
        f1  = float(f1_score(y_true, pred)),
        auc = float(roc_auc_score(y_true, proba)),
        brier = float(bri),
        cm = cm.tolist()
    )

# -------------------------- models ---------------------------

def train_lr_rf(Xh_train, Xh_test, y_train, y_test):
    lr = LogisticRegression(max_iter=1000, n_jobs=None, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    lr.fit(Xh_train, y_train)
    rf.fit(Xh_train, y_train)

    lr_proba = lr.predict_proba(Xh_test)[:,1]; lr_pred = (lr_proba >= 0.5).astype(int)
    rf_proba = rf.predict_proba(Xh_test)[:,1]; rf_pred = (rf_proba >= 0.5).astype(int)

    lr_m = eval_and_plot("LR", y_test, lr_proba, lr_pred)
    rf_m = eval_and_plot("RF", y_test, rf_proba, rf_pred)

    joblib.dump(lr, ART/"model_lr.pkl")
    joblib.dump(rf, ART/"model_rf.pkl")
    return lr_m, rf_m

def train_mlp(Xh_train, Xh_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                        solver="adam", max_iter=500, random_state=42)
    mlp.fit(Xh_train, y_train)
    proba = mlp.predict_proba(Xh_test)[:,1]
    pred  = (proba >= 0.5).astype(int)
    m = eval_and_plot("MLP", y_test, proba, pred)
    joblib.dump(mlp, ART/"model_mlp.pkl")
    return m, mlp

def train_transformer(Xh_train, Xh_test, y_train, y_test, epochs=10,
                      d_model=64, n_heads=4, n_layers=2, ff_mult=4,
                      batch_size=64, lr=1e-3, seed=42):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        print(f"[Transformer] PyTorch not available; skipping. ({e})")
        return None

    torch.manual_seed(seed)

    Xtr = to_dense(Xh_train).astype(np.float32)
    Xte = to_dense(Xh_test).astype(np.float32)
    ytr = y_train.to_numpy().astype(np.int64)
    yte = y_test.to_numpy().astype(np.int64)
    n_features = Xtr.shape[1]

    class FeatureTokenizer(nn.Module):
        def __init__(self, n_features, d_model):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
            self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        def forward(self, x):               # x: [B, D]
            return x.unsqueeze(-1) * self.weight + self.bias   # [B, D, d]

    class TinyTransformer(nn.Module):
        def __init__(self, n_features, d_model, n_heads, n_layers, ff_mult):
            super().__init__()
            self.tok = FeatureTokenizer(n_features, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*ff_mult,
                batch_first=True, activation="gelu", dropout=0.0
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.cls  = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2)
            )
        def forward(self, x):               # x: [B, D]
            toks = self.tok(x)              # [B, D, d]
            h = self.enc(toks).mean(dim=1)  # mean pool
            h = self.norm(h)
            return self.cls(h)

    model = TinyTransformer(n_features, d_model, n_heads, n_layers, ff_mult)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()

    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    # train
    model.train()
    for ep in range(1, epochs+1):
        loss_ep = 0.0
        for xb, yb in dl_tr:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_ep += loss.item() * xb.size(0)
        if ep % 5 == 0:
            print(f"[Transformer] epoch {ep:02d}/{epochs}  loss={loss_ep/len(ds_tr):.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte))
        proba  = torch.softmax(logits, 1)[:, 1].numpy()
        pred   = logits.argmax(1).numpy()

    m = eval_and_plot("TF", yte, proba, pred)
    try:
        import torch
        torch.save(model.state_dict(), ART/"transformer.pt")
    except Exception:
        pass
    return m

# ------------------------ robustness checks ------------------------

def cv_f1_mlp(Xh_train, y_train):
    Xh_tr_dense = to_dense(Xh_train)
    f1s = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(Xh_tr_dense, y_train):
        clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400, random_state=42)
        clf.fit(Xh_tr_dense[tr], y_train.iloc[tr])
        pred = clf.predict(Xh_tr_dense[va])
        f1s.append(f1_score(y_train.iloc[va], pred))
    # bar plot
    plt.figure(figsize=(5,3.6))
    plt.bar(range(1,6), f1s, color="steelblue")
    plt.ylim(0.95,1.01)
    plt.axhline(y=np.mean(f1s), color="red", linestyle="--", label=f"Mean F1={np.mean(f1s):.3f}")
    plt.xlabel("Fold"); plt.ylabel("F1"); plt.title("5-Fold CV (MLP)"); plt.legend()
    savefig("40_robustness_cv")
    return np.array(f1s)

def noise_robustness_proxy(Xh_test, y_test, sigma=0.05):
    Xd = to_dense(Xh_test).copy()
    from sklearn.linear_model import LogisticRegression
    lr_proxy = LogisticRegression(max_iter=1000, random_state=0).fit(Xd, y_test)
    acc_clean = (lr_proxy.predict(Xd) == y_test).mean()
    acc_noise = (lr_proxy.predict(Xd + np.random.normal(0, sigma, size=Xd.shape)) == y_test).mean()
    # line plot
    plt.figure(figsize=(5,3.6))
    xs = [0.0, 0.02, sigma, 0.10]
    ys = []
    for s in xs:
        ys.append((lr_proxy.predict(Xd + np.random.normal(0, s, size=Xd.shape)) == y_test).mean())
    plt.plot(xs, ys, marker="o", color="darkorange"); plt.ylim(0.95,1.01)
    plt.xlabel("Noise level (σ)"); plt.ylabel("Accuracy"); plt.title("Noise Robustness")
    savefig("41_robustness_noise")
    return float(acc_noise), float(acc_clean)

# --------------------------- batch scoring -------------------------

def batch_predict(csv_path):
    """Load artifacts and score a CSV of new patients."""
    preprocessor = joblib.load(ART/"preprocessor.pkl")
    mlp = joblib.load(ART/"model_mlp.pkl")
    centers = np.load(ART/"ifcm_centers.npy")

    df = pd.read_csv(csv_path)
    # match the training cleaning as best as possible
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"?":np.nan,"nan":np.nan})

    X = preprocessor.transform(df)

    # compute cluster id using stored centers
    def euclid2(a,b):
        a2=(a*a).sum(1,keepdims=True); b2=(b*b).sum(1,keepdims=True).T
        return np.maximum(a2+b2-2*a@b.T,1e-12)

    Xd = to_dense(X)
    D = euclid2(Xd, centers)
    U = (1.0/D)**(1.0/(2.0-1.0))
    U = U/U.sum(axis=1,keepdims=True)
    cl = U.argmax(1)

    Xh = hstack_feature(X, cl)
    proba = mlp.predict_proba(Xh)[:,1]
    pred  = (proba >= 0.5).astype(int)

    out = df.copy()
    out["cluster"] = cl
    out["probability"] = proba
    out["prediction"] = pred
    out.to_csv(ART/"scored_output.csv", index=False)
    print(f"Saved results to {ART/'scored_output.csv'}")
    print(out[["cluster","probability","prediction"]].head())
    return out

# ------------------------------ main ------------------------------

def main(args):
    if args.score:
        return batch_predict(args.score)

    print("=== CKD Pipeline (IFCM C=3, Hybrid MLP + Transformer, Augmentation-Correct) ===")

    # 1) Load & clean
    df = load_and_clean("kidney_disease.csv")
    print(f"[1/9] Loaded dataset: {df.shape}")

    # 2) Real split (test remains real-only)
    y = df["classification"]; X = df.drop(columns=["classification"])
    pre_raw, num_cols, cat_cols = make_preprocessor(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"[2/9] REAL split: train {X_tr.shape}, test {X_te.shape}")
    print("NOTE: Train/Test are split BEFORE any augmentation; test is REAL-ONLY.")

    # 3) Augment train only (bootstrap)
    syn = synth_bootstrap_train(X_tr, y_tr, num_cols, cat_cols, size_multiplier=0.8)
    syn["source"] = "synthetic"
    print("[3/9] Synthetic data (bootstrap on TRAIN ONLY)…")

    # 4) Combine (with source flag) and EDA
    real = pd.concat([X_tr.reset_index(drop=True), y_tr.reset_index(drop=True).rename("classification")], axis=1)
    real["source"] = "real"
    train_combined = pd.concat([real, syn], ignore_index=True)
    # Quick EDA on combined set (before encoding)
    eda_after_combine(train_combined)

    Xc_tr_raw = train_combined.drop(columns=["classification","source"])
    yc_tr     = train_combined["classification"]
    Xc_te_raw = X_te.copy(); yc_te = y_te.copy()

    # 5) Encode (fit on combined train → no leakage to test)
    pre, _, _ = make_preprocessor(Xc_tr_raw)
    Xc_tr = pre.fit_transform(Xc_tr_raw)
    Xc_te = pre.transform(Xc_te_raw)
    joblib.dump(pre, ART/"preprocessor.pkl")
    print(f"[5/9] Encoded shapes: train_combined {Xc_tr.shape} | test_real {Xc_te.shape}")

    # 6) IFCM (C=3) and append cluster_id
    print("[6/9] Incremental FCM (C=3) …")
    Xh_tr, Xh_te, cl_tr, cl_te = run_ifcm_and_append(Xc_tr, Xc_te)
    np.save(ART/"train_cluster_ids.npy", cl_tr)
    np.save(ART/"test_cluster_ids.npy",  cl_te)

    # 7) Baselines + Hybrid MLP
    print("[7/9] Training LR/RF/MLP …")
    lr_m, rf_m = train_lr_rf(Xh_tr, Xh_te, yc_tr, yc_te)
    mlp_m, mlp_model = train_mlp(Xh_tr, Xh_te, yc_tr, yc_te)

    # 8) Transformer (optional)
    print("[8/9] Training Transformer (attention) …")
    tf_m = train_transformer(Xh_tr, Xh_te, yc_tr, yc_te, epochs=10)

    # 9) Robustness (CV + noise)
    print("[9/9] Robustness checks …")
    f1s = cv_f1_mlp(Xh_tr, yc_tr)
    noise_acc, _ = noise_robustness_proxy(Xh_te, yc_te, sigma=0.05)
    print(f"    5-Fold F1 (MLP): {np.round(f1s,3).tolist()}  mean={f1s.mean():.3f} ± {f1s.std():.3f}")
    print(f"    Accuracy under noise perturbation (σ=0.05): {noise_acc:.3f}")

    # Summaries & artifacts
    results = {
        "LR": lr_m, "RF": rf_m, "MLP": mlp_m, "TF": tf_m,
        "cv_f1_mean": float(f1s.mean()), "cv_f1_std": float(f1s.std()),
        "noise_acc_sigma_0.05": float(noise_acc)
    }
    with open(ART/"metrics_latest.json","w") as f: json.dump(results, f, indent=2)

    md = [
        "# Results Summary (Hybrid + IFCM C=3)",
        "",
        f"- **MLP**  ACC={mlp_m['acc']:.3f}, F1={mlp_m['f1']:.3f}, ROC-AUC={mlp_m['auc']:.3f}, Brier={mlp_m['brier']:.3f}",
        f"- **LR**   ACC={lr_m['acc']:.3f}, F1={lr_m['f1']:.3f}, ROC-AUC={lr_m['auc']:.3f}",
        f"- **RF**   ACC={rf_m['acc']:.3f}, F1={rf_m['f1']:.3f}, ROC-AUC={rf_m['auc']:.3f}",
    ]
    if tf_m is not None:
        md.append(f"- **Transformer** ACC={tf_m['acc']:.3f}, F1={tf_m['f1']:.3f}, ROC-AUC={tf_m['auc']:.3f}, Brier={tf_m['brier']:.3f}")
    md += [
        "",
        f"- **5-fold CV (MLP)**: mean F1 = {f1s.mean():.3f} ± {f1s.std():.3f}",
        f"- **Noise robustness (σ=0.05)**: acc = {noise_acc:.3f}",
        "",
        "_Note: test set is real-only; synthetic rows augment train only._",
        f"_All figures saved under `plots1/`._"
    ]
    write_md(ROOT/"results.md", "\n".join(md))

    print("\nDONE. See ./plots1, ./artifacts and results.md")
    print("NOTE: Real test is isolated from augmentation to avoid leakage/optimism.")

# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=str, default=None,
                    help="CSV file of new patient rows to score (uses saved artifacts).")
    args = ap.parse_args()
    main(args)
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap

# ------------ config ------------
PLOTS_DIR = "plots1"  # change if needed

models = ["lr", "rf", "mlp", "tf"]          # order they’ll appear
nice_name = {"lr": "Logistic Regression",
             "rf": "Random Forest",
             "mlp": "MLP",
             "tf": "Transformer"}

# expected canonical filenames in your folder (based on your screenshot)
canon = {
    "confusion": {
        "lr": "lr_confusion_matrix.png",
        "rf": "rf_confusion_matrix.png",
        "mlp": "mlp_confusion_matrix.png",
        "tf": "tf_confusion_matrix.png"
    },
    "roc": {
        "lr": "lr_roc_curve.png",
        "rf": "rf_roc_curve.png",
        "mlp": "mlp_roc_curve.png",
        "tf": "tf_roc_curve.png"
    },
    "pr": {
        "lr": "lr_pr_curve.png",
        "rf": "rf_pr_curve.png",
        "mlp": "mlp_pr_curve.png",
        "tf": "tf_pr_curve.png"
    },
    "calibration": {
        "lr": "lr_calibration.png",
        "rf": "rf_calibration.png",
        "mlp": "mlp_calibration.png",
        "tf": "tf_calibration.png"
    }
}

# Optional: fallback fuzzy matching if the canonical name isn’t there
def find_image_fuzzy(dirpath, patterns):
    """
    Try to glob a list of patterns in order, returning first match or None.
    """
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(dirpath, pat)))
        if hits:
            return hits[0]
    return None

def load_img_or_placeholder(path, title):
    """
    Load an image if it exists; otherwise return None and the title will show a warning.
    """
    if path and os.path.exists(path):
        try:
            return mpimg.imread(path)
        except Exception:
            return None
    return None

def panel(ax, img, title, missing_note=None):
    ax.axis("off")
    ax.set_title("\n".join(wrap(title, 40)), fontsize=11)
    if img is not None:
        ax.imshow(img)
    else:
        # draw placeholder text if missing
        ax.text(0.5, 0.5, missing_note or "MISSING",
                color="crimson", ha="center", va="center", fontsize=12,
                transform=ax.transAxes)

# Make a single 2×2 grid for one metric
def make_grid(metric_key, outname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    axes = axes.ravel()

    missing_files = []

    for i, m in enumerate(models):
        title = nice_name[m]
        # canonical path
        wanted = os.path.join(PLOTS_DIR, canon[metric_key][m])

        # if not found, try fuzzy patterns
        if not os.path.exists(wanted):
            base = f"{m}"
            # fuzzy patterns that often occur
            pats = [
                f"{base}_{metric_key}*.png",          # e.g., lr_roc*.png
                f"{base}*{metric_key}*.png",          # e.g., lr_*roc*.png
                f"*{base}*{metric_key}*.png"          # extra-safe
            ]
            wanted = find_image_fuzzy(PLOTS_DIR, pats)

        img = load_img_or_placeholder(wanted, title)
        if img is None:
            missing_files.append(canon[metric_key][m])
            panel(axes[i], None, title, missing_note=f"(MISSING: {canon[metric_key][m]})")
        else:
            panel(axes[i], img, title)

    fig.suptitle({
        "confusion": "Confusion Matrices — All Models",
        "roc": "ROC Curves — All Models",
        "pr": "PR Curves — All Models",
        "calibration": "Calibration (Reliability) Curves — All Models"
    }[metric_key], fontsize=14)

    plt.savefig(outname, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved {metric_key} grid → {outname}")
    if missing_files:
        print("   Some files were missing:", ", ".join(missing_files))
    else:
        print("   All panels populated.")

# ------------ build all four grids ------------
os.makedirs(PLOTS_DIR, exist_ok=True)

make_grid("confusion", "combined_confusion_matrices_all.png")
make_grid("roc",        "combined_roc_curves_all.png")
make_grid("pr",         "combined_pr_curves_all.png")
make_grid("calibration","combined_calibration_all.png")

print("NOTE: Real test is isolated from augmentation to avoid leakage/optimism.")
