#!/usr/bin/env python3
"""
Scaling law study: RMSE vs M for 5D Poisson with vanilla ELM.

Fix Q_int = max(M_VALUES) to keep overdetermined.
Q_bnd = Q_int // 10 per face.
R_m = 0.1 (optimal for 5D).

Output: log-log plot of RMSE vs M with best-fit line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy.stats import linregress

from locelm import compute_basis_nd, init_params_nd

jax.config.update("jax_enable_x64", True)

DIM = 5
LO = jnp.zeros(DIM)
HI = jnp.ones(DIM)
R_M = 0.1
SEED = 0


def exact_solution(x):
    return jnp.sum(jnp.sin(jnp.pi / 2 * x))


def source_fn(x):
    return (jnp.pi**2) / 4 * jnp.sum(jnp.sin(jnp.pi / 2 * x))


def sample_boundary(key, n_per_face):
    keys = jax.random.split(key, 2 * DIM)
    all_pts = []
    for i in range(DIM):
        for side, val in enumerate([0.0, 1.0]):
            pts = jax.random.uniform(keys[2 * i + side], (n_per_face, DIM))
            pts = pts.at[:, i].set(val)
            all_pts.append(pts)
    return jnp.concatenate(all_pts, axis=0)


def run_experiment(M, Q_int, seed=SEED):
    Q_bnd_per_face = max(Q_int // 10, 100)
    key = jax.random.PRNGKey(seed)
    k_params, k_int, k_bnd, k_eval = jax.random.split(key, 4)

    W, b = init_params_nd(k_params, M, R_M, DIM)

    x_int = jax.random.uniform(k_int, (Q_int, DIM))
    x_bnd = sample_boundary(k_bnd, Q_bnd_per_face)

    V_int, lap_V_int = compute_basis_nd(x_int, W, b, LO, HI)
    V_bnd = compute_basis_nd(x_bnd, W, b, LO, HI)[0]

    A = jnp.concatenate([-lap_V_int, V_bnd], axis=0)
    rhs = jnp.concatenate([jax.vmap(source_fn)(x_int), jax.vmap(exact_solution)(x_bnd)])

    beta, _, _, _ = scipy.linalg.lstsq(
        np.array(A), np.array(rhs), lapack_driver="gelsd"
    )

    x_eval = jax.random.uniform(k_eval, (10000, DIM))
    V_eval = compute_basis_nd(x_eval, W, b, LO, HI)[0]
    u_num = V_eval @ jnp.array(beta)
    u_exact = jax.vmap(exact_solution)(x_eval)

    return float(jnp.sqrt(jnp.mean((u_num - u_exact) ** 2)))


def main():
    M_VALUES = [100, 200, 500, 1000, 2000, 3000]
    Q_INT = max(M_VALUES) * 4

    print("5D Poisson scaling law: -Δu = f on [0,1]^5")
    print(f"Q_int={Q_INT}, R_m={R_M}, seed={SEED}")
    print(f"\n{'M':>6}  {'Q_bnd/face':>10}  {'RMSE':>12}")
    print("-" * 36)

    results = []
    for M in M_VALUES:
        rms = run_experiment(M, Q_INT)
        results.append((M, rms))
        print(f"{M:>6}  {max(Q_INT // 10, 100):>10}  {rms:>12.3e}")

    M_arr = np.array([m for m, _ in results], dtype=float)
    R_arr = np.array([r for _, r in results], dtype=float)

    slope, intercept, r_value, _, _ = linregress(np.log10(M_arr), np.log10(R_arr))
    print(f"\nRMSE ∝ M^{slope:.2f},  R={r_value:.4f},  R²={r_value**2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        M_arr,
        R_arr,
        marker="o",
        color="C0",
        linestyle="-",
        linewidth=1.4,
        markersize=7,
        label="data",
    )

    M_fit = np.logspace(np.log10(M_arr.min()), np.log10(M_arr.max()), 200)
    ax.loglog(
        M_fit,
        10 ** (slope * np.log10(M_fit) + intercept),
        linestyle="--",
        color="C0",
        linewidth=1.2,
        label=f"fit: RMSE∝M$^{{{slope:.2f}}}$, R={r_value:.3f}",
    )

    ax.set_xlabel("M (basis functions)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(
        f"Scaling law — 5D Poisson  ($R_m$={R_M}, $Q_{{int}}$={Q_INT})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = Path(__file__).parent / "scaling_law_5d.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
