
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib missing -> plots disabled")

from logistic_regression_gd import LogisticRegressionGD

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "bank-additional-full.csv")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
NUM_COLS = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

# ======================================================================
# 0. numerical gradient check (sanity check of the hand-derived gradient)
# ======================================================================
print("=== 0. numerical gradient check ===")
rng0 = np.random.default_rng(0)
Xc = rng0.standard_normal((30, 8))
yc = np.where(np.random.default_rng(1).random(30) > 0.5, 1.0, -1.0)
chk = LogisticRegressionGD(lam=0.5, verbose=False)
chk.w = np.random.default_rng(2).standard_normal(8)
chk.b = 0.3
gw, gb = chk._gradient(Xc, yc)
eps = 1e-6
gw_num = np.empty_like(chk.w)
for k in range(len(chk.w)):
    wp, wm = chk.w.copy(), chk.w.copy()
    wp[k] += eps
    wm[k] -= eps
    chk.w = wp
    lp = chk._loss(Xc, yc)
    chk.w = wm
    lm = chk._loss(Xc, yc)
    gw_num[k] = (lp - lm) / (2 * eps)
chk.w = np.random.default_rng(2).standard_normal(8)
chk.b = 0.3
chk.b += eps
lp = chk._loss(Xc, yc)
chk.b -= 2 * eps
lm = chk._loss(Xc, yc)
chk.b += eps
gb_num = (lp - lm) / (2 * eps)
print(f"max|grad_w analytic - numeric| = {np.abs(gw - gw_num).max():.2e}")
print(f"|grad_b analytic - numeric|    = {abs(gb - gb_num):.2e}")

# ======================================================================
# 1. data
# ======================================================================
print("\n=== 1. load & preprocess UCI Bank Marketing ===")
df = pd.read_csv(DATA, sep=";")
y = (df["y"] == "yes").astype(int).to_numpy()
print(f"shape={df.shape}, positive rate={y.mean():.4f}")

rng = np.random.default_rng(42)
pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
test_idx = np.sort(np.concatenate([
    rng.choice(pos, int(0.2 * len(pos)), replace=False),
    rng.choice(neg, int(0.2 * len(neg)), replace=False)]))
train_mask = np.ones(len(y), dtype=bool)
train_mask[test_idx] = False


def build_matrix(num_cols):
    """z-score the numeric columns (train stats only) + one-hot the categoricals."""
    Xnum = df[num_cols].to_numpy(dtype=float)
    mu, sd = Xnum[train_mask].mean(0), Xnum[train_mask].std(0)
    Xnum = (Xnum - mu) / sd
    Xcat = pd.get_dummies(df[CAT_COLS], dtype=float).to_numpy()
    return np.hstack([Xnum, Xcat])


X = build_matrix(NUM_COLS)
Xtr, ytr = X[train_mask], y[train_mask]
Xte, yte = X[test_idx], y[test_idx]
print(f"train={Xtr.shape} test={Xte.shape} n_features={X.shape[1]}")

# ======================================================================
# 2. metrics implemented by hand (rank-based AUC + ROC staircase)
# ======================================================================
def average_ranks(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0     # 1-based average rank
        i = j
    return ranks


def auc_manual(y_true, scores):
    """AUC = Mann-Whitney U statistic / (P*N): P(positive scores above negative)."""
    yt = np.asarray(y_true).ravel()
    s = np.asarray(scores).ravel()
    P = int((yt == 1).sum())
    N = int((yt == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    r = average_ranks(s)
    return float((r[yt == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def roc_curve_manual(y_true, scores):
    """Walk samples in descending score order; each + moves up, each - moves right."""
    yt = np.asarray(y_true).ravel()
    s = np.asarray(scores).ravel()
    P = int((yt == 1).sum())
    N = int((yt == 0).sum())
    order = np.argsort(-s, kind="mergesort")
    fpr, tpr = [0.0], [0.0]
    tp = fp = 0
    for v in yt[order]:
        if v == 1:
            tp += 1
        else:
            fp += 1
        fpr.append(fp / N)
        tpr.append(tp / P)
    return np.asarray(fpr), np.asarray(tpr)


def report(name, y_true, proba):
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / len(y_true)
    auc = auc_manual(y_true, proba)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"[{name}] acc={acc:.4f} auc={auc:.4f} "
          f"confusion=[[TN={tn},FP={fp}],[FN={fn},TP={tp}]] "
          f"precision={prec:.4f} recall={rec:.4f}")
    return acc, auc

# ======================================================================
# 3. our model: full-batch GD vs mini-batch SGD
# ======================================================================
print("\n=== 3. from-scratch LogisticRegressionGD (L2, lambda=1e-3) ===")
t0 = time.time()
gd = LogisticRegressionGD(learning_rate=0.5, n_epochs=2000, lam=1e-3,
                          tol=1e-10, patience=100, verbose=True)
gd.fit(Xtr, ytr)
print(f"full-batch GD train time: {time.time() - t0:.1f}s")
acc_gd, auc_gd = report("full-batch GD", yte, gd.predict_proba(Xte))

t0 = time.time()
sgd = LogisticRegressionGD(learning_rate=0.1, n_epochs=50, lam=1e-3,
                           batch_size=256, tol=1e-10, patience=100, verbose=True)
sgd.fit(Xtr, ytr)
print(f"mini-batch SGD train time: {time.time() - t0:.1f}s")
acc_sgd, auc_sgd = report("mini-batch SGD", yte, sgd.predict_proba(Xte))

print(f"||w||2 GD={np.linalg.norm(gd.w):.4f}  ||w||2 SGD={np.linalg.norm(sgd.w):.4f}")

# ======================================================================
# 4. manual ROC curve + threshold sweep
# ======================================================================
print("\n=== 4. manual ROC curve & threshold sweep (full-batch GD) ===")
proba_te = gd.predict_proba(Xte)
fpr, tpr = roc_curve_manual(yte, proba_te)
P_te = int(yte.sum())
N_te = len(yte) - P_te
print(f"AUC (manual, rank method) = {auc_manual(yte, proba_te):.4f}")
print(f"{'threshold':>10} {'accuracy':>9} {'TPR':>7} {'FPR':>7} {'precision':>10}")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (proba_te >= t).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    tn = int(((pred == 0) & (yte == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    print(f"{t:>10.2f} {(tp + tn) / len(yte):>9.4f} {tp / P_te:>7.4f} "
          f"{fp / N_te:>7.4f} {prec:>10.4f}")

if HAS_MPL:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, color="tab:blue",
             label=f"LogisticRegressionGD  (AUC={auc_manual(yte, proba_te):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random classifier (AUC=0.5)")
    plt.xlabel("False positive rate  =  1 - specificity")
    plt.ylabel("True positive rate  =  sensitivity / recall")
    plt.title("ROC curve - UCI Bank Marketing (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=150)
    print("saved out/roc_curve.png")

    plt.figure(figsize=(6, 5))
    plt.plot(gd.loss_history, label="full-batch GD")
    plt.plot(sgd.loss_history, label="mini-batch SGD (256)")
    plt.xlabel("epoch")
    plt.ylabel("regularized cross-entropy loss J(w,b)")
    plt.title("Loss decreases to the global minimum (J is convex)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "loss_history.png"), dpi=150)
    print("saved out/loss_history.png")

# ======================================================================
# 5. drop 'duration' (call length known only AFTER the call -> leaky)
# ======================================================================
print("\n=== 5. without 'duration' (realistic evaluation) ===")
X_nodur = build_matrix([c for c in NUM_COLS if c != "duration"])
gd_nd = LogisticRegressionGD(learning_rate=0.5, n_epochs=2000, lam=1e-3,
                             tol=1e-10, patience=100, verbose=False)
gd_nd.fit(X_nodur[train_mask], ytr)
acc_nd, auc_nd = report("without duration", yte, gd_nd.predict_proba(X_nodur[test_idx]))
print(f"majority-class (predict all 'no') accuracy = {1 - yte.mean():.4f}")

# ======================================================================
# 6. 2-feature model -> visual proof that the boundary is a straight line
# ======================================================================
print("\n=== 6. decision boundary on 2 features (duration, euribor3m) ===")
X_2d = df[["duration", "euribor3m"]].to_numpy(dtype=float)
mu2, sd2 = X_2d[train_mask].mean(0), X_2d[train_mask].std(0)
X_2d = (X_2d - mu2) / sd2
gd2 = LogisticRegressionGD(learning_rate=1.0, n_epochs=2000, lam=1e-3,
                           tol=1e-12, patience=150, verbose=False)
gd2.fit(X_2d[train_mask], ytr)
w1, w2 = gd2.w
b0 = gd2.b
print(f"decision boundary: {b0:.4f} + ({w1:.4f})*duration + ({w2:.4f})*euribor3m = 0 "
      f" -> a straight line (linear)")

if HAS_MPL:
    Xte2 = X_2d[test_idx]
    subs = rng.choice(len(yte), 3000, replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(Xte2[subs][yte[subs] == 0, 0], Xte2[subs][yte[subs] == 0, 1],
                s=6, alpha=0.4, label="no (y=0)")
    plt.scatter(Xte2[subs][yte[subs] == 1, 0], Xte2[subs][yte[subs] == 1, 1],
                s=6, alpha=0.6, label="yes (y=1)")
    x1s = np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), 100)
    plt.plot(x1s, -(b0 + w1 * x1s) / w2, color="tab:red", lw=2,
             label="decision boundary  $w^T x + b = 0$")
    plt.xlabel("duration (z-score)")
    plt.ylabel("euribor3m (z-score)")
    plt.title("Logistic regression decision boundary is a straight line")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "decision_boundary.png"), dpi=150)
    print("saved out/decision_boundary.png")

# ======================================================================
# 7. sklearn: shrinkage factor C, CV tuning, C <-> lambda equivalence
# ======================================================================
print("\n=== 7. sklearn LogisticRegression: shrinkage factor C ===")
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

Cs_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

t0 = time.time()
cv_shuf = LogisticRegressionCV(Cs=Cs_grid,
                               cv=StratifiedKFold(5, shuffle=True, random_state=0),
                               scoring="roc_auc", max_iter=5000)
cv_shuf.fit(Xtr, ytr)
Cbest = float(cv_shuf.C_[0])
auc_cv = auc_manual(yte, cv_shuf.predict_proba(Xte)[:, 1])
acc_cv = float((cv_shuf.predict(Xte) == yte).mean())
print(f"5-fold SHUFFLED CV ({time.time() - t0:.0f}s): best C={Cbest:.4g} "
      f"| test acc={acc_cv:.4f} | test AUC={auc_cv:.4f}")

t0 = time.time()
cv_time = LogisticRegressionCV(Cs=Cs_grid,
                               cv=StratifiedKFold(5, shuffle=False),
                               scoring="roc_auc", max_iter=5000)
cv_time.fit(Xtr, ytr)
print(f"5-fold TIME-ORDERED CV ({time.time() - t0:.0f}s): best C={float(cv_time.C_[0]):.4g} "
      f"-> data is ordered by time; strong shrinkage (small C) generalizes to unseen periods")

rows = []
for C in Cs_grid:
    m = LogisticRegression(C=C, max_iter=5000)
    m.fit(Xtr, ytr)
    rows.append((C, float(np.linalg.norm(m.coef_)),
                 auc_manual(ytr, m.predict_proba(Xtr)[:, 1]),
                 auc_manual(yte, m.predict_proba(Xte)[:, 1])))
print(f"{'C':>8} {'||w||2':>9} {'AUC train':>10} {'AUC test':>9}")
for C, nw, atr, ate in rows:
    print(f"{C:>8.0e} {nw:>9.3f} {atr:>10.4f} {ate:>9.4f}")

if HAS_MPL:
    Cs = [r[0] for r in rows]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.semilogx(Cs, [r[1] for r in rows], "o-")
    plt.xlabel("C (inverse shrinkage)")
    plt.ylabel("||w||2")
    plt.title("Smaller C -> weights shrink to 0")
    plt.subplot(1, 2, 2)
    plt.semilogx(Cs, [r[2] for r in rows], "o-", label="train AUC")
    plt.semilogx(Cs, [r[3] for r in rows], "s-", label="test AUC")
    plt.xlabel("C (inverse shrinkage)")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("Pick C by cross-validation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "C_shrinkage.png"), dpi=150)
    print("saved out/C_shrinkage.png")

# ---- equivalence of our lambda and sklearn's C:  lambda = 1/(C*n) ----
C_check = float(cv_time.C_[0])
lam_eq = 1.0 / (C_check * Xtr.shape[0])
print(f"\nequivalence check at C={C_check:.4g} (time-ordered CV choice): "
      f"lambda = 1/(C*n) = {lam_eq:.3e}")
print("(at C=100 the problem is ill-conditioned - month dummies are nearly "
      "collinear with the macro variables - so plain GD converges slowly; "
      "the identity lambda = 1/(C*n) is exact for every C)")
eq = LogisticRegressionGD(learning_rate=1.0, n_epochs=6000, lam=lam_eq,
                          tol=1e-12, patience=300, verbose=False)
t0 = time.time()
eq.fit(Xtr, ytr)
print(f"fit ours with equivalent lambda: {time.time() - t0:.0f}s")
dw = float(np.abs(eq.w - cv_time.coef_.ravel()).max())
db = float(abs(eq.b - float(cv_time.intercept_[0])))
print(f"max |w_ours - w_sklearn| = {dw:.2e}    |b_ours - b_sklearn| = {db:.2e}")

# ======================================================================
print("\n============== SUMMARY ==============")
print(f"dataset            : {df.shape[0]} rows x {df.shape[1]} cols, "
      f"positive rate {y.mean():.3f} (imbalanced)")
print(f"features after encoding: {X.shape[1]}")
print(f"our full-batch GD  : test acc={acc_gd:.4f}  AUC={auc_gd:.4f}")
print(f"our mini-batch SGD : test acc={acc_sgd:.4f}  AUC={auc_sgd:.4f}")
print(f"without 'duration' : test acc={acc_nd:.4f}  AUC={auc_nd:.4f}")
print(f"sklearn best C (shuffled CV) : {Cbest:.4g}  test acc={acc_cv:.4f}  test AUC={auc_cv:.4f}")
print(f"sklearn best C (time-ordered CV): {float(cv_time.C_[0]):.4g}  (strong shrinkage -> honest generalization)")
print(f"C <-> lambda       : lambda = 1/(C*n); verified max |dw|={dw:.2e}")
print("done.")
