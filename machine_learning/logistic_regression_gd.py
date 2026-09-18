# -*- coding: utf-8 -*-
"""
logistic_regression_gd.py
==========================
Regularized (L2) logistic regression written from scratch with numpy.
Training is (full-batch) gradient descent or (mini-batch) stochastic
gradient descent. No machine-learning library is used.

Model
-----
    P(y = 1 | x) = sigmoid(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))

Objective (labels encoded as y in {+1, -1})
--------------------------------------------
    J(w, b) = (1/n) * sum_i log(1 + exp(-y_i (w^T x_i + b)))   (cross-entropy)
              + (lambda / 2) * ||w||^2                        (L2 penalty)

The intercept b is deliberately NOT regularized (standard practice).

Gradient (chain rule)
---------------------
    d J / d w = -(1/n) X^T [ y .* sigmoid(-y .* (X w + b)) ] + lambda * w
    d J / d b = -(1/n) sum [ y .* sigmoid(-y .* (X w + b)) ]

Because J is convex (see answers.md, Q1), gradient descent converges to
the *global* minimum.
"""
import numpy as np


class LogisticRegressionGD:
    """Binary regularized logistic regression trained by gradient descent."""

    def __init__(self, learning_rate=0.5, n_epochs=2000, lam=1e-3,
                 batch_size=None, tol=1e-10, patience=100, seed=0,
                 verbose=True):
        """
        Parameters
        ----------
        learning_rate : float   step size of gradient descent
        n_epochs      : int     maximum number of epochs (full passes)
        lam           : float   L2 regularization strength on w (not on b)
        batch_size    : int or None
                                None -> full-batch gradient descent
                                k    -> mini-batch SGD with batches of size k
        tol           : float   loss-improvement threshold for early stopping
        patience      : int     stop after this many epochs without improvement
        seed          : int     RNG seed for mini-batch shuffling
        verbose       : bool    print a training summary
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.lam = lam
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.w = None                # weights, shape (d,)
        self.b = None                # intercept, scalar
        self.loss_history = []       # full-data loss after every epoch

    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(z):
        z = np.clip(np.asarray(z, dtype=float), -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, X, y):
        """J(w, b) on the plain feature matrix X (no bias column); y in {+1, -1}."""
        z = y * (X @ self.w + self.b)
        # log(1 + exp(-z)) computed stably with logaddexp
        data_loss = np.mean(np.logaddexp(0.0, -z))
        reg = 0.5 * self.lam * float(self.w @ self.w)   # b excluded
        return float(data_loss) + reg

    def _gradient(self, X, y):
        """Analytic gradient of J; X is the plain feature matrix (no bias column)."""
        z = y * (X @ self.w + self.b)
        g = y * self.sigmoid(-z)                        # shape (m,)
        grad_w = -(X.T @ g) / X.shape[0] + self.lam * self.w
        grad_b = -float(g.mean())
        return grad_w, grad_b

    def _step(self, X, y):
        gw, gb = self._gradient(X, y)
        self.w -= self.learning_rate * gw
        self.b -= self.learning_rate * gb

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y).ravel() == 1, 1.0, -1.0)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        rng = np.random.default_rng(self.seed)

        best, patience_left = np.inf, self.patience
        for epoch in range(1, self.n_epochs + 1):
            if self.batch_size is None:                # full-batch GD
                self._step(X, y)
            else:                                      # one full pass of mini-batch SGD
                perm = rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    self._step(X[idx], y[idx])

            loss = self._loss(X, y)                   # monitor on ALL data
            self.loss_history.append(loss)
            if loss < best - self.tol:
                best, patience_left = loss, self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if self.verbose:
            print(f"[LR-GD] stopped after {len(self.loss_history)} epochs, "
                  f"loss={best:.6f}, ||w||2={float(np.linalg.norm(self.w)):.4f}")
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X):
        """w^T x + b  (the log-odds)."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict_proba(self, X):
        """P(y = 1 | x)."""
        return self.sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
