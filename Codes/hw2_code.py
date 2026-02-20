#!/usr/bin/env python3
"""
DSC 240 Homework 2 (Q5-Q7) - Linear regression solvers

This script implements:
  - Closed-form least-squares and ridge regression
  - Batch gradient descent (GD)
  - Stochastic / mini-batch gradient descent (SGD)

It then (when the .npy datasets are available) will:
  Q5: Compare GD/SGD/Closed-form solutions via L2 distance (lsqr and ridge)
  Q6: Plot learning curves for lsqr and ridge under GD vs SGD
  Q7: Plot learning curves for ridge SGD with batch sizes 1, 10, 100

Expected dataset filenames in --data_dir:
  - data_X_Q5Q6.npy
  - data_y_Q5Q6.npy
  - data_X_Q7.npy
  - data_y_Q7.npy
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Objectives
# ----------------------------
def objective_lsqr(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Least-squares objective:
        E(w) = (1/N) ||Xw - y||^2
    """
    N = X.shape[0]
    r = X @ w - y
    return float((r @ r) / N)


def objective_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """
    Ridge objective:
        E(w) = (1/N) ||Xw - y||^2 + lam ||w||^2
    """
    return objective_lsqr(X, y, w) + float(lam * (w @ w))


# ----------------------------
# Closed-form solutions
# ----------------------------
def closed_form_lsqr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form least squares:
        w* = (X^T X)^(-1) X^T y
    Uses a pseudo-inverse for numerical stability.
    """
    return np.linalg.pinv(X) @ y


def closed_form_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge (matches HW derivation for E2(w)=(1/N)||Xw-y||^2 + lam||w||^2):
        w* = (X^T X + N*lam*I)^(-1) X^T y
    """
    N, d = X.shape
    A = X.T @ X + (N * lam) * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


# ----------------------------
# Gradient-based solvers
# ----------------------------
def _lipschitz_constant(X: np.ndarray, lam: float) -> float:
    """
    Lipschitz constant of the gradient of the ridge objective:
        grad = (2/N) X^T (Xw - y) + 2 lam w

    L = 2/N * ||X||_2^2 + 2 lam
    """
    N = X.shape[0]
    # Spectral norm; uses SVD internally.
    s = np.linalg.norm(X, 2)
    return float((2.0 / N) * (s ** 2) + 2.0 * lam)


@dataclass
class SolverResult:
    w: np.ndarray
    obj_history: List[float]


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
    lr: Optional[float] = None,
    max_iters: int = 2000,
    tol: Optional[float] = None,
) -> SolverResult:
    """
    Batch gradient descent on ridge objective.
    If lam=0, this is least-squares GD.
    """
    N, d = X.shape
    w = np.zeros(d, dtype=float)

    if lr is None:
        L = _lipschitz_constant(X, lam)
        lr = 1.0 / L

    obj_hist: List[float] = []
    prev_obj: Optional[float] = None

    for t in range(max_iters):
        # Full gradient
        r = X @ w - y
        grad = (2.0 / N) * (X.T @ r) + 2.0 * lam * w
        w = w - lr * grad

        obj = objective_ridge(X, y, w, lam)
        obj_hist.append(obj)

        if tol is not None and prev_obj is not None:
            if abs(prev_obj - obj) < tol:
                break
        prev_obj = obj

    return SolverResult(w=w, obj_history=obj_hist)


def sgd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
    lr: Optional[float] = None,
    max_iters: int = 20000,
    batch_size: int = 1,
    seed: int = 0,
    tol: Optional[float] = None,
) -> SolverResult:
    """
    Stochastic / mini-batch gradient descent on ridge objective.

    Uses the unbiased mini-batch gradient estimate for the data term:
        grad_data = (2/m) X_B^T (X_B w - y_B),
    then adds exact reg gradient:
        grad_reg = 2 lam w.

    NOTE: We still evaluate and record the *full* objective using all data points
    at each iteration (as required by the assignment learning curves).
    """
    N, d = X.shape
    w = np.zeros(d, dtype=float)

    if lr is None:
        # SGD typically needs a smaller step than batch GD for stability.
        L = _lipschitz_constant(X, lam)
        lr = 0.5 / L

    rng = np.random.default_rng(seed)
    obj_hist: List[float] = []
    prev_obj: Optional[float] = None

    for t in range(max_iters):
        idx = rng.choice(N, size=batch_size, replace=False if batch_size <= N else True)
        Xb = X[idx]
        yb = y[idx]

        rb = Xb @ w - yb
        grad_data = (2.0 / batch_size) * (Xb.T @ rb)
        grad = grad_data + 2.0 * lam * w

        w = w - lr * grad

        obj = objective_ridge(X, y, w, lam)
        obj_hist.append(obj)

        if tol is not None and prev_obj is not None:
            if abs(prev_obj - obj) < tol:
                break
        prev_obj = obj

    return SolverResult(w=w, obj_history=obj_hist)


# ----------------------------
# Utilities
# ----------------------------
def l2_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    return float(np.linalg.norm(w1 - w2))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_learning_curves(
    curves: Dict[str, List[float]],
    title: str,
    out_path: str,
) -> None:
    plt.figure()
    for label, hist in curves.items():
        plt.plot(range(1, len(hist) + 1), hist, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Objective E(w)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main experiments for Q5-Q7
# ----------------------------
def run_q5_q6(data_dir: str, out_dir: str, lam: float, gd_iters: int, sgd_iters: int) -> None:
    X_path = os.path.join(data_dir, "data_X_Q5Q6.npy")
    y_path = os.path.join(data_dir, "data_y_Q5Q6.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print("[Q5/Q6] Missing data_X_Q5Q6.npy and/or data_y_Q5Q6.npy in:", data_dir)
        return

    X = np.load(X_path).astype(float)
    y = np.load(y_path).astype(float).reshape(-1)

    _ensure_dir(out_dir)

    # ---------------- Q5: least squares (no regularization) ----------------
    print("\n=== Q5: Least squares (lambda=0) ===")
    w_cf_ls = closed_form_lsqr(X, y)
    res_gd_ls = gradient_descent(X, y, lam=0.0, max_iters=gd_iters)
    res_sgd_ls = sgd(X, y, lam=0.0, max_iters=sgd_iters, batch_size=1)

    print("L2(GD, SGD)        =", l2_distance(res_gd_ls.w, res_sgd_ls.w))
    print("L2(GD, ClosedForm) =", l2_distance(res_gd_ls.w, w_cf_ls))
    print("L2(SGD, ClosedForm)=", l2_distance(res_sgd_ls.w, w_cf_ls))

    # ---------------- Q5: ridge (with regularization) ----------------
    print("\n=== Q5: Ridge (lambda={:.1e}) ===".format(lam))
    w_cf_r = closed_form_ridge(X, y, lam=lam)
    res_gd_r = gradient_descent(X, y, lam=lam, max_iters=gd_iters)
    res_sgd_r = sgd(X, y, lam=lam, max_iters=sgd_iters, batch_size=1)

    print("L2(GD, SGD)        =", l2_distance(res_gd_r.w, res_sgd_r.w))
    print("L2(GD, ClosedForm) =", l2_distance(res_gd_r.w, w_cf_r))
    print("L2(SGD, ClosedForm)=", l2_distance(res_sgd_r.w, w_cf_r))

    # ---------------- Q6: learning curves ----------------
    print("\n=== Q6: Saving learning curves to", out_dir, "===")
    plot_learning_curves(
        curves={"GD": res_gd_ls.obj_history, "SGD (batch=1)": res_sgd_ls.obj_history},
        title="Q6: Least Squares Objective vs Iteration",
        out_path=os.path.join(out_dir, "q6_lsqr_gd_vs_sgd.png"),
    )
    plot_learning_curves(
        curves={"GD": res_gd_r.obj_history, "SGD (batch=1)": res_sgd_r.obj_history},
        title="Q6: Ridge Objective vs Iteration (lambda={:.1e})".format(lam),
        out_path=os.path.join(out_dir, "q6_ridge_gd_vs_sgd.png"),
    )


def run_q7(data_dir: str, out_dir: str, lam: float, sgd_iters: int) -> None:
    X_path = os.path.join(data_dir, "data_X_Q7.npy")
    y_path = os.path.join(data_dir, "data_y_Q7.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print("[Q7] Missing data_X_Q7.npy and/or data_y_Q7.npy in:", data_dir)
        return

    X = np.load(X_path).astype(float)
    y = np.load(y_path).astype(float).reshape(-1)

    _ensure_dir(out_dir)

    print("\n=== Q7: Mini-batch SGD for ridge (lambda={:.1e}) ===".format(lam))
    for bs in [1, 10, 100]:
        bs_eff = min(bs, X.shape[0])
        res = sgd(X, y, lam=lam, max_iters=sgd_iters, batch_size=bs_eff, seed=0)
        out_path = os.path.join(out_dir, f"q7_ridge_sgd_batch{bs_eff}.png")
        plot_learning_curves(
            curves={f"SGD (batch={bs_eff})": res.obj_history},
            title=f"Q7: Ridge SGD Learning Curve (batch={bs_eff}, lambda={lam:.1e})",
            out_path=out_path,
        )
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing the .npy datasets")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save plots")
    parser.add_argument("--lambda_reg", type=float, default=1e-6, help="Ridge regularization parameter lambda")
    parser.add_argument("--gd_iters", type=int, default=2000, help="Max iterations for batch GD")
    parser.add_argument("--sgd_iters", type=int, default=20000, help="Max iterations for SGD/mini-batch SGD")
    args = parser.parse_args()

    run_q5_q6(args.data_dir, args.out_dir, args.lambda_reg, args.gd_iters, args.sgd_iters)
    run_q7(args.data_dir, args.out_dir, args.lambda_reg, args.sgd_iters)


if __name__ == "__main__":
    main()
