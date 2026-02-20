"""DSC 240 (Winter 2026) - Homework 1 Code Submission

This script contains the Python code used for HW1 questions that require code:
- Q7: Eigenvectors of a 2x2 matrix (given hand-computed eigenvalues)
- Q8: Plot decision regions for a linear classifier in 2D
- Q9: Generate linearly separable data + run Perceptron Learning Algorithm + plots

Run:
    python3 HW1_code.py

Outputs:
    q8_combined.png
    q9_b_seed0_n20.png
    q9_c_seed1_n20.png
    q9_d_seed2_n100.png
    q9_e_seed3_n1000.png

Dependencies:
    numpy, matplotlib, sympy
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Sympy is used only for Q7 eigenvector computation (allowed per HW spec).
import sympy as sp


# Save all output files (plots) next to this script, regardless of where it is run from.
OUTDIR = Path(__file__).resolve().parent


# -----------------------------
# Q7: Eigenvectors (given eigenvalues)
# -----------------------------

def q7_eigenvectors() -> None:
    """Compute eigenvectors for A = [[1,4],[2,3]] given eigenvalues 5 and -1.

    The eigenvalues are assumed to be computed by hand (as required).
    We only use Python to compute eigenvectors.
    """

    A = sp.Matrix([[1, 4], [2, 3]])

    # Hand-computed eigenvalues (do NOT compute them with Python per instructions).
    lam1 = 5
    lam2 = -1

    v1 = (A - lam1 * sp.eye(2)).nullspace()[0]
    v2 = (A - lam2 * sp.eye(2)).nullspace()[0]

    print("Q7: Eigenvectors (given eigenvalues 5 and -1)")
    print(f"A =\n{A}")
    print(f"lambda = {lam1}, eigenvector basis = {v1}")
    print(f"lambda = {lam2}, eigenvector basis = {v2}")
    print()


# -----------------------------
# Q8: Linear classifier plots
# -----------------------------


def _decision_scores(w: np.ndarray, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute w0 + w1*x1 + w2*x2 over a grid."""
    w0, w1, w2 = float(w[0]), float(w[1]), float(w[2])
    return w0 + w1 * X1 + w2 * X2


def _plot_one_linear_classifier(ax: plt.Axes, w: np.ndarray, title: str,
                               xlim: Tuple[float, float] = (-5, 5),
                               ylim: Tuple[float, float] = (-5, 5),
                               grid_n: int = 401) -> None:
    """Plot decision regions and boundary line for h(x)=sign(w^T x)."""
    x1 = np.linspace(xlim[0], xlim[1], grid_n)
    x2 = np.linspace(ylim[0], ylim[1], grid_n)
    X1, X2 = np.meshgrid(x1, x2)

    scores = _decision_scores(w, X1, X2)
    preds = np.sign(scores)

    # Fill regions.
    ax.contourf(X1, X2, preds, levels=[-1, 0, 1], alpha=0.2)

    # Boundary where score == 0.
    w0, w1, w2 = float(w[0]), float(w[1]), float(w[2])
    if abs(w2) > 1e-12:
        boundary = -(w0 + w1 * x1) / w2
        ax.plot(x1, boundary, linewidth=2)
    elif abs(w1) > 1e-12:
        # Vertical line
        ax.axvline(-w0 / w1, linewidth=2)
    else:
        # Degenerate: w1=w2=0 => constant classifier; no boundary
        pass

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def q8_make_plots() -> None:
    """Generate the Q8 plot for w=[1,2,3]^T and w=-[1,2,3]^T."""
    w_pos = np.array([1.0, 2.0, 3.0])
    w_neg = -w_pos

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    _plot_one_linear_classifier(axes[0], w_pos, "w=[1,2,3]^T")
    _plot_one_linear_classifier(axes[1], w_neg, "w=-[1,2,3]^T")

    fig.suptitle("Q8: Decision regions for h(x)=sign(w^T x)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "q8_combined.png", dpi=200)
    plt.close(fig)

    print(f"Q8: wrote {(OUTDIR / 'q8_combined.png').as_posix()}")
    print()


# -----------------------------
# Q9: Perceptron experiments
# -----------------------------


def GenerateData(margin: float, number: int, seed: Optional[int] = None) -> List[Tuple[np.ndarray, int]]:
    """Generate a linearly separable dataset using the HW-provided rule.

    Returns a list of (point, label) where point has shape (2,1) and label in {0,1}.

    Notes:
    - Class 0 if x1 + x2 > margin
    - Class 1 if x1 + x2 < -margin
    """
    if seed is not None:
        np.random.seed(seed)

    data: List[Tuple[np.ndarray, int]] = []
    i = 0
    while i < number:
        point = np.random.randn(2, 1)
        s = float(point[0, 0] + point[1, 0])
        if s - margin > 0:
            data.append((point, 0))
            i += 1
        elif s + margin < 0:
            data.append((point, 1))
            i += 1

    return data


def perceptron_train(data: List[Tuple[np.ndarray, int]], max_updates: int = 200000) -> Tuple[np.ndarray, int]:
    """Train perceptron on data.

    We use augmented feature x=[1,x1,x2]^T and map labels:
        label 0 -> y=+1
        label 1 -> y=-1

    Update rule (standard perceptron):
        if y*(w·x) <= 0: w <- w + y*x

    Returns:
        (w, num_updates)
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for p, lab in data:
        x_aug = np.array([1.0, float(p[0, 0]), float(p[1, 0])], dtype=float)
        y = 1 if lab == 0 else -1
        X_list.append(x_aug)
        y_list.append(y)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)

    w = np.zeros(3, dtype=float)
    updates = 0

    while updates < max_updates:
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * float(np.dot(w, xi)) <= 0.0:
                w = w + yi * xi
                updates += 1
                mistakes += 1
                if updates >= max_updates:
                    break
        if mistakes == 0:
            break

    return w, updates


def q9_plot_result(data: List[Tuple[np.ndarray, int]], w: np.ndarray, filename: str, title: str) -> None:
    """Plot data points, target boundary x2=-x1, and perceptron boundary."""

    pts = np.array([p.flatten() for p, _lab in data], dtype=float)  # (n,2)
    labs = np.array([lab for _p, lab in data], dtype=int)

    pad = 1.0
    x_min, x_max = float(pts[:, 0].min() - pad), float(pts[:, 0].max() + pad)
    y_min, y_max = float(pts[:, 1].min() - pad), float(pts[:, 1].max() + pad)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    ax.scatter(pts[labs == 0, 0], pts[labs == 0, 1], marker='o', label='class 0')
    ax.scatter(pts[labs == 1, 0], pts[labs == 1, 1], marker='x', label='class 1')

    xs = np.linspace(x_min, x_max, 200)
    ax.plot(xs, -xs, linewidth=2, label='target f: x2=-x1')

    w0, w1, w2 = float(w[0]), float(w[1]), float(w[2])
    if abs(w2) > 1e-12:
        ax.plot(xs, -(w0 + w1 * xs) / w2, '--', linewidth=2, label='perceptron g')
    elif abs(w1) > 1e-12:
        ax.axvline(-w0 / w1, linestyle='--', linewidth=2, label='perceptron g')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTDIR / filename, dpi=200)
    plt.close(fig)


def q9_run_all() -> None:
    """Run the full set of Q9 experiments required in the writeup."""

    margin = 0.2

    # (b) n=20, seed=0
    data20_seed0 = GenerateData(margin, 20, seed=0)
    w20_seed0, upd20_seed0 = perceptron_train(data20_seed0)
    q9_plot_result(data20_seed0, w20_seed0, "q9_b_seed0_n20.png", f"Q9(b): n=20, seed=0, updates={upd20_seed0}")

    # (c) n=20, seed=1
    data20_seed1 = GenerateData(margin, 20, seed=1)
    w20_seed1, upd20_seed1 = perceptron_train(data20_seed1)
    q9_plot_result(data20_seed1, w20_seed1, "q9_c_seed1_n20.png", f"Q9(c): n=20, seed=1, updates={upd20_seed1}")

    # (d) n=100, seed=2
    data100_seed2 = GenerateData(margin, 100, seed=2)
    w100_seed2, upd100_seed2 = perceptron_train(data100_seed2)
    q9_plot_result(data100_seed2, w100_seed2, "q9_d_seed2_n100.png", f"Q9(d): n=100, seed=2, updates={upd100_seed2}")

    # (e) n=1000, seed=3
    data1000_seed3 = GenerateData(margin, 1000, seed=3)
    w1000_seed3, upd1000_seed3 = perceptron_train(data1000_seed3)
    q9_plot_result(data1000_seed3, w1000_seed3, "q9_e_seed3_n1000.png", f"Q9(e): n=1000, seed=3, updates={upd1000_seed3}")

    print(f"Q9: wrote {(OUTDIR / 'q9_b_seed0_n20.png').as_posix()}")
    print(f"Q9: wrote {(OUTDIR / 'q9_c_seed1_n20.png').as_posix()}")
    print(f"Q9: wrote {(OUTDIR / 'q9_d_seed2_n100.png').as_posix()}")
    print(f"Q9: wrote {(OUTDIR / 'q9_e_seed3_n1000.png').as_posix()}")
    print()

    print("Perceptron update counts:")
    print(f"  (b) n=20,  seed=0:  updates={upd20_seed0}")
    print(f"  (c) n=20,  seed=1:  updates={upd20_seed1}")
    print(f"  (d) n=100, seed=2:  updates={upd100_seed2}")
    print(f"  (e) n=1000,seed=3:  updates={upd1000_seed3}")
    print()


def main() -> None:
    q7_eigenvectors()
    q8_make_plots()
    q9_run_all()


if __name__ == "__main__":
    main()
