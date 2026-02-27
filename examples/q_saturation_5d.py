#!/usr/bin/env python3
"""
Q-saturation study for 5D Poisson with vanilla ELM.

Fix M (total basis functions), vary Q_int (interior collocation points).
Q_bnd = Q_int // 10 per face (10 faces total, so boundary total = Q_int).

Output: 1x3 subplot figure for M ∈ {500, 1000, 2000}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

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
    Q_bnd_per_face = max(Q_int // 10, 50)
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
    M_VALUES = [500, 1000, 2000]
    Q_VALUES = [500, 1000, 2000, 5000, 10000, 20000]

    all_results = {}

    for M in M_VALUES:
        print(f"\n===== M={M} =====")
        print(f"{'Q_int':>8}  {'Q_bnd/face':>10}  {'RMSE':>12}")
        print("-" * 36)
        pts = []
        for Q in Q_VALUES:
            rms = run_experiment(M, Q)
            pts.append((Q, rms))
            print(f"{Q:>8}  {max(Q // 10, 50):>10}  {rms:>12.3e}")
        all_results[M] = pts
        print()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for idx, M in enumerate(M_VALUES):
        ax = axes[idx]
        pts = all_results[M]
        Q_arr = np.array([q for q, _ in pts], dtype=float)
        R_arr = np.array([r for _, r in pts], dtype=float)

        ax.loglog(
            Q_arr,
            R_arr,
            marker="o",
            color="C0",
            linestyle="-",
            linewidth=1.4,
            markersize=6,
        )

        ax.axvline(
            x=M,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Q_int=M={M}",
        )
        ax.set_xlabel("$Q_{int}$ (interior collocation)", fontsize=11)
        ax.set_title(f"M = {M}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("RMSE", fontsize=11)
    fig.suptitle(
        f"Q-saturation — 5D Poisson  ($R_m$={R_M}, seed={SEED})",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / "q_saturation_5d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
