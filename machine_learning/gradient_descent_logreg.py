# -*- coding: utf-8 -*-
"""
UCI Bank Marketing dataset -- hand-written gradient descent for
L2-regularized logistic regression
=====================================================================
Data source : https://archive.ics.uci.edu/dataset/222/bank+marketing
Data file   : data/bank-additional-full.csv (41188 rows, 20 features + target y)

Everything below is implemented from scratch (model part uses only NumPy /
pandas / matplotlib, no sklearn):
  1. Data loading & preprocessing: one-hot encoding + z-score standardization
     + stratified train/validation/test split
  2. Model: L2-regularized logistic regression (intercept is NOT regularized)
        loss:     J(w) = -(1/n) sum [ y_i log sigmoid(x_i.w) + (1-y_i) log(1-sigmoid(x_i.w)) ]
                          + (lambda/2) ||w||^2            (intercept w0 excluded)
        gradient: grad J(w) = (1/n) X^T (sigmoid(Xw) - y) + lambda * w (no reg. on intercept)
  3. Optimization (hand-written):
       - fixed-step gradient descent: step = 1/L, L = 0.25 * sigma_max^2(X^T X / n) + lambda
         (theoretically guaranteed descent step; sigma_max estimated by power iteration)
       - Armijo backtracking line search (auto-shrinks step, guarantees sufficient decrease)
  4. Finite-difference gradient check to validate the analytic gradient
  5. Metrics implemented by hand: accuracy / precision / recall / F1 / AUC / log-loss
  6. Comparison against sklearn LogisticRegression as a correctness check
  7. Outputs: loss curves, ROC curve, weight plot, metrics.txt report
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")          # headless environment, save figures only
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ global config
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
DROP_DURATION = True           # duration=0 leaks the target (no call -> no purchase); drop by default
LAMBDAS = [0.0, 1e-4, 1e-2, 1.0]   # regularization strengths to compare
MAX_ITER = 5000
TOL = 1e-8                     # convergence threshold on the infinity-norm of the gradient


# ------------------------------------------------------------------ 1. data loading & preprocessing
# Numeric columns of this dataset are fixed (the rest are categorical);
# pandas 3.x may read quoted columns as strings, so convert them explicitly.
KNOWN_NUMERIC = {"age", "campaign", "pdays", "previous", "emp.var.rate",
                 "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
                 "duration"}


def load_and_preprocess(path: Path, drop_duration: bool = True):
    """Read the CSV, one-hot encode categorical columns, return (X, y, numeric columns)."""
    df = pd.read_csv(path, sep=";")
    y = (df["y"] == "yes").astype(int).to_numpy()
    X = df.drop(columns=["y"])

    if drop_duration:
        X = X.drop(columns=["duration"])

    num_cols = [c for c in X.columns if c in KNOWN_NUMERIC]
    cat_cols = [c for c in X.columns if c not in KNOWN_NUMERIC]
    for c in num_cols:                       # pandas 3.x may keep them as strings -> force numeric
        X[c] = pd.to_numeric(X[c])
    print(f"  categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"  numeric features ({len(num_cols)}): {num_cols}")

    X = pd.get_dummies(X, columns=cat_cols)     # one-hot encoding
    print(f"  feature dim after one-hot: {X.shape[1]}")
    return X, y, num_cols


def stratified_split(y: np.ndarray, ratios=(0.6, 0.2, 0.2), seed=SEED):
    """Stratified split into train/val/test; returns three row-index arrays."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for k in (0, 1):
        idx = rng.permutation(np.where(y == k)[0])
        n = len(idx)
        b0, b1 = round(n * ratios[0]), round(n * (ratios[0] + ratios[1]))
        parts[0].append(idx[:b0])
        parts[1].append(idx[b0:b1])
        parts[2].append(idx[b1:])
    return [np.sort(np.concatenate(p)) for p in parts]


def standardize(X_train, X_val, X_test, num_cols):
    """z-score standardization using TRAIN statistics only (no leakage)."""
    for X in (X_train, X_val, X_test):       # pandas 3.x forbids assigning floats into int64 columns
        X[num_cols] = X[num_cols].astype(float)
    mu = X_train[num_cols].mean()
    sd = X_train[num_cols].std().replace(0, 1.0)
    for X in (X_train, X_val, X_test):
        X.loc[:, num_cols] = (X[num_cols] - mu) / sd


# ------------------------------------------------------------------ 2. model: hand-written L2 logistic regression
class LogisticRegressionGD:
    """L2-regularized logistic regression solved with gradient descent
    (fixed step size or Armijo backtracking)."""

    def __init__(self, l2: float = 0.0, fit_intercept: bool = True):
        self.l2 = float(l2)
        self.fit_intercept = fit_intercept

    # ---- math core ----
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z = np.clip(z, -700.0, 700.0)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1))]) if self.fit_intercept else X

    def _reg_mask(self, w: np.ndarray) -> np.ndarray:
        """Regularization coefficient vector: lambda for features, 0 for the intercept."""
        if self.fit_intercept:
            return np.r_[np.full(w.shape[0] - 1, self.l2), 0.0]
        return np.full(w.shape[0], self.l2)

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """J(w) = -(1/n) sum[y log sig + (1-y) log(1-sig)] + (lambda/2) ||w||^2"""
        z = X @ w
        ll = -np.mean(y * z - np.logaddexp(0.0, z))       # logaddexp avoids overflow
        reg = 0.5 * float(np.sum(w * w * self._reg_mask(w)))
        return ll + reg

    def grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """grad J(w) = (1/n) X^T (sigmoid(Xw) - y) + lambda * w (no reg. on intercept)."""
        p = self.sigmoid(X @ w)
        return X.T @ (p - y) / X.shape[0] + self._reg_mask(w) * w

    # ---- optimization ----
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 1.0,
            max_iter: int = MAX_ITER, tol: float = TOL,
            backtracking: bool = True, alpha: float = 0.3, beta: float = 0.5,
            verbose: bool = False) -> "LogisticRegressionGD":
        """
        Gradient-descent training.
        backtracking=True  : Armijo line search, adapts the step each round so that
                             f(w - a*g) <= f(w) - c*a*||g||^2 (sufficient decrease);
        backtracking=False : plain gradient descent with a fixed step (should be 1/L).
        """
        Xb = self._add_bias(X)
        w = np.zeros(Xb.shape[1])                       # convex problem: zero init is fine
        self.loss_history = []
        self.stop_reason = "max_iter"
        f_prev = np.inf
        t0 = time.time()

        for it in range(max_iter):
            f = self.loss(w, Xb, y)
            g = self.grad(w, Xb, y)
            self.loss_history.append(f)

            if np.linalg.norm(g, ord=np.inf) < tol:     # criterion 1: gradient ~ 0
                self.stop_reason = "grad_tol"
                if verbose:
                    print(f"    converged at iter {it}, ||g||_inf = {np.linalg.norm(g, ord=np.inf):.2e}")
                break
            if abs(f_prev - f) < 1e-12:                 # criterion 2: loss stopped decreasing
                self.stop_reason = "loss_flat"          # (near-separable data: grad-norm decays slowly)
                if verbose:
                    print(f"    converged at iter {it}, loss change < 1e-12")
                break
            f_prev = f

            step = lr
            if backtracking:                            # Armijo condition: f(w-a*g) <= f(w) - c*a*||g||^2
                g_sq = float(g @ g)
                for _ in range(80):
                    if self.loss(w - step * g, Xb, y) <= f - alpha * step * g_sq:
                        break
                    step *= beta

            w = w - step * g

        self.w = w
        self.coef_ = w[:-1] if self.fit_intercept else w
        self.intercept_ = float(w[-1]) if self.fit_intercept else 0.0
        self.n_iter_ = it + 1
        self.fit_time_ = time.time() - t0
        return self

    # ---- prediction ----
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        return Xb @ self.w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def lipschitz_constant(X: np.ndarray, l2: float) -> float:
    """L = 0.25 * sigma_max^2(X^T X / n) + lambda; sigma_max via power iteration."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(200):
        u = X.T @ (X @ v)
        n = np.linalg.norm(u)
        if n == 0:
            break
        v = u / n
    sigma2 = float(v @ (X.T @ (X @ v))) / X.shape[0]     # largest eigenvalue
    return 0.25 * sigma2 + l2


def check_gradient(model: LogisticRegressionGD, X: np.ndarray, y: np.ndarray):
    """Finite-difference check of the analytic gradient (on a small subsample)."""
    rng = np.random.default_rng(1)
    idx = rng.choice(len(y), size=min(200, len(y)), replace=False)
    Xs, ys = model._add_bias(X[idx]), y[idx]
    w = rng.standard_normal(Xs.shape[1])
    g_ana = model.grad(w, Xs, ys)
    g_num = np.empty_like(g_ana)
    eps = 1e-6
    for j in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        g_num[j] = (model.loss(wp, Xs, ys) - model.loss(wm, Xs, ys)) / (2 * eps)
    err = np.linalg.norm(g_ana - g_num) / np.linalg.norm(g_ana + g_num)
    print(f"  gradient check (finite difference) relative error: {err:.3e}  ->  "
          f"{'PASS' if err < 1e-6 else 'FAIL!'}")
    return err


# ------------------------------------------------------------------ 3. evaluation metrics (hand-written)
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])


def metrics(y_true, y_pred, scores):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    p = np.clip(scores, 1e-12, 1 - 1e-12)
    logloss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    auc = roc_auc(y_true, scores)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "auc": auc, "logloss": logloss}, cm


def roc_auc(y_true, scores):
    """Hand-written AUC: TPR/FPR at every score threshold, trapezoid under ROC."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return float("nan")
    tpr, fpr = [], []
    for t in np.unique(s):
        pred = (s >= t).astype(int)
        tpr.append(np.sum((pred == 1) & (y == 1)) / P)
        fpr.append(np.sum((pred == 1) & (y == 0)) / N)
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


# ------------------------------------------------------------------ 4. plots
def plot_loss_curves(runs: dict, path: Path):
    """Training loss curves for different lambda values (log y-axis)."""
    plt.figure(figsize=(9, 6))
    for label, hist in runs.items():
        plt.semilogy(hist, lw=1.8, label=label)
    plt.xlabel("iteration")
    plt.ylabel("loss J(w) (log scale)")
    plt.title("Gradient descent convergence (different regularization lambda)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc(my_model, sk_model, Xte, yte, path: Path):
    """Test-set ROC curve: hand-written model vs sklearn."""
    plt.figure(figsize=(7, 7))
    for name, scorer in (("hand-written GD logistic regression", my_model),
                         ("sklearn LogisticRegression", sk_model)):
        s = scorer.predict_proba(Xte)
        if s.ndim > 1:
            s = s[:, 1]
        fpr, tpr = [], []
        for t in np.unique(s):
            pred = (s >= t).astype(int)
            tpr.append(np.sum((pred == 1) & (yte == 1)) / yte.sum())
            fpr.append(np.sum((pred == 1) & (yte == 0)) / (len(yte) - yte.sum()))
        fpr = np.r_[0.0, fpr, 1.0]
        tpr = np.r_[0.0, tpr, 1.0]
        order = np.argsort(fpr)
        auc = np.trapezoid(tpr[order], fpr[order])
        plt.plot(fpr[order], tpr[order], lw=2, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve on the test set")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_top_weights(model, feature_names, path: Path, topk: int = 15):
    """Top-k feature weights by absolute value."""
    order = np.argsort(np.abs(model.coef_))[::-1][:topk]
    names = np.array(feature_names)[order]
    vals = model.coef_[order]
    plt.figure(figsize=(9, 6))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]
    plt.barh(range(topk), vals[::-1], color=colors[::-1])
    plt.yticks(range(topk), names[::-1], fontsize=9)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("weight w")
    plt.title(f"Top-{topk} feature weights by |w| (lambda={model.l2:g})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------------ 5. main
def main():
    print("=" * 72)
    print("1. Data loading & preprocessing")
    print("=" * 72)
    X_df, y, num_cols = load_and_preprocess(DATA_PATH, DROP_DURATION)
    print(f"  total samples: {len(y)}, positives (y=yes): {y.sum()} ({y.mean()*100:.1f}%)")

    tr_idx, va_idx, te_idx = stratified_split(y)
    Xtr_df, Xva_df, Xte_df = (X_df.iloc[i].reset_index(drop=True) for i in (tr_idx, va_idx, te_idx))
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
    print(f"  split: train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)}")

    standardize(Xtr_df, Xva_df, Xte_df, num_cols)      # train statistics only

    Xtr = Xtr_df.to_numpy(dtype=np.float64)   # one-hot (bool) mixed with numerics -> float64
    Xva = Xva_df.to_numpy(dtype=np.float64)
    Xte = Xte_df.to_numpy(dtype=np.float64)
    feature_names = list(Xtr_df.columns)

    # theoretically guaranteed fixed step: 1 / L
    L = lipschitz_constant(Xtr, 0.0)
    lr_fixed = 1.0 / L
    print(f"  Lipschitz constant L = {L:.4f}, fixed step 1/L = {lr_fixed:.4f}")

    print("=" * 72)
    print("2. Gradient check (finite difference)")
    print("=" * 72)
    probe = LogisticRegressionGD(l2=0.1)
    check_gradient(probe, Xtr, ytr)

    print("=" * 72)
    print("3. Gradient-descent training: lambda comparison (Armijo backtracking)")
    print("=" * 72)
    runs_backtrack, models, val_metrics = {}, {}, {}
    best_lam, best_auc = None, -1.0

    for lam in LAMBDAS:
        model = LogisticRegressionGD(l2=lam).fit(
            Xtr, ytr, backtracking=True, max_iter=MAX_ITER, tol=TOL)
        models[lam] = model
        runs_backtrack[f"lambda={lam:g}"] = model.loss_history

        m_va, _ = metrics(yva, model.predict(Xva), model.predict_proba(Xva))
        m_tr, _ = metrics(ytr, model.predict(Xtr), model.predict_proba(Xtr))
        val_metrics[lam] = m_va
        print(f"  lambda={lam:<6g} | iters={model.n_iter_:<4}({model.stop_reason}) | "
              f"train acc={m_tr['acc']:.4f} auc={m_tr['auc']:.4f} | "
              f"val   acc={m_va['acc']:.4f} auc={m_va['auc']:.4f} f1={m_va['f1']:.4f} | "
              f"time {model.fit_time_:.2f}s")
        if m_va["auc"] > best_auc:
            best_auc, best_lam = m_va["auc"], lam

    print(f"\n  >>> best lambda by validation AUC: {best_lam:g} (AUC = {best_auc:.4f})")

    # fixed-step gradient descent as a comparison against backtracking
    model_fixed = LogisticRegressionGD(l2=best_lam).fit(
        Xtr, ytr, lr=lr_fixed, backtracking=False, max_iter=MAX_ITER, tol=TOL)
    print(f"  [baseline] fixed-step 1/L = {lr_fixed:.4f} gradient descent: "
          f"iters={model_fixed.n_iter_}, final loss={model_fixed.loss_history[-1]:.6f}")

    print("=" * 72)
    print("4. Test-set evaluation (best lambda) vs sklearn")
    print("=" * 72)
    best_model = models[best_lam]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as sk_auc
    C = 1.0 / best_lam if best_lam > 0 else 1e6     # sklearn: C = 1/lambda
    sk_model = LogisticRegression(C=C, max_iter=5000, tol=1e-8).fit(Xtr, ytr)

    for name, model in (("hand-written GD (backtracking)", best_model),
                        ("sklearn                    ", sk_model)):
        s = model.predict_proba(Xte)
        if s.ndim > 1:
            s = s[:, 1]
        m_te, cm = metrics(yte, (s >= 0.5).astype(int), s)
        print(f"  [{name}] | test acc={m_te['acc']:.4f} prec={m_te['prec']:.4f} "
              f"rec={m_te['rec']:.4f} f1={m_te['f1']:.4f} auc={m_te['auc']:.4f} "
              f"logloss={m_te['logloss']:.4f}")
        print(f"          confusion matrix [[TN,FP],[FN,TP]] = {cm.tolist()}")

    # cross-check the hand-written AUC against sklearn's
    my_auc = metrics(yte, (best_model.predict_proba(Xte) >= .5).astype(int),
                     best_model.predict_proba(Xte))[0]["auc"]
    print(f"  [check] hand-written AUC={my_auc:.4f} vs sklearn AUC="
          f"{sk_auc(yte, sk_model.predict_proba(Xte)[:, 1]):.4f}")

    print("=" * 72)
    print("5. Save plots & report")
    print("=" * 72)
    # convergence curves: Armijo backtracking for every lambda + fixed-step baseline
    runs_all = dict(runs_backtrack)
    runs_all[f"lambda={best_lam:g} (fixed step 1/L)"] = model_fixed.loss_history
    plot_loss_curves(runs_all, OUT_DIR / "loss_curves.png")
    plot_roc(best_model, sk_model, Xte, yte, OUT_DIR / "roc_curve.png")
    plot_top_weights(best_model, feature_names, OUT_DIR / "top_weights.png")
    print("  saved: output/loss_curves.png, output/roc_curve.png, output/top_weights.png")

    # export weights
    wdf = pd.DataFrame({"feature": feature_names, "weight": best_model.coef_})
    wdf = wdf.reindex(wdf.weight.abs().sort_values(ascending=False).index)
    wdf.to_csv(OUT_DIR / "best_weights.csv", index=False)

    # text report
    lines = [
        "UCI Bank Marketing -- hand-written gradient descent + L2 logistic regression",
        "=" * 68,
        f"data: bank-additional-full.csv ({len(y)} samples, positives {y.sum()}, {y.mean()*100:.2f}%)",
        f"features: {Xtr.shape[1]} dims after one-hot (duration dropped: {DROP_DURATION})",
        f"split: train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)}",
        f"Lipschitz constant L = {L:.4f}, fixed step 1/L = {lr_fixed:.4f}",
        "",
        "lambda selection (Armijo backtracking gradient descent, validation set):",
    ]
    for lam in LAMBDAS:
        m = val_metrics[lam]
        lines.append(f"  lambda={lam:<6g}: val acc={m['acc']:.4f} f1={m['f1']:.4f} "
                     f"auc={m['auc']:.4f} logloss={m['logloss']:.4f} "
                     f"(iters={models[lam].n_iter_}, {models[lam].fit_time_:.2f}s)")
    lines += [
        "",
        f"best lambda = {best_lam:g} (validation AUC = {best_auc:.4f})",
        "",
        "test-set comparison:",
    ]
    for name, model in (("hand-written GD (Armijo backtracking)", best_model),
                        ("sklearn LogisticRegression", sk_model)):
        s = model.predict_proba(Xte)
        if s.ndim > 1:
            s = s[:, 1]
        m_te, cm = metrics(yte, (s >= 0.5).astype(int), s)
        lines.append(f"  {name}: acc={m_te['acc']:.4f} prec={m_te['prec']:.4f} "
                     f"rec={m_te['rec']:.4f} f1={m_te['f1']:.4f} auc={m_te['auc']:.4f}")
        lines.append(f"    confusion matrix [[TN,FP],[FN,TP]] = {cm.tolist()}")
    lines += [
        "",
        f"fixed-step (1/L) gradient descent: iters={model_fixed.n_iter_}, "
        f"final loss={model_fixed.loss_history[-1]:.6f}",
        "",
        "top-10 feature weights (by |w|):",
    ]
    for f, v in zip(wdf.head(10).feature, wdf.head(10).weight):
        lines.append(f"  {f:<30s} {v:+.4f}")
    lines.append(f"\n  intercept w0 = {best_model.intercept_:.4f}")
    lines.append("\ngenerated: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    report = "\n".join(lines)
    (OUT_DIR / "metrics.txt").write_text(report, encoding="utf-8")
    print("  saved: output/metrics.txt, output/best_weights.csv")
    print("\n" + report)


if __name__ == "__main__":
    main()
