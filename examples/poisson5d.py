#!/usr/bin/env python3
"""
5D Poisson with vanilla ELM.

PDE: -Δu = f(x),  x ∈ [0,1]^5
f(x) = (π²/4) Σᵢ sin(πxᵢ/2)
u_exact(x) = Σᵢ sin(πxᵢ/2)
Dirichlet BC: u = u_exact on ∂Ω

From PINNacle benchmark (PoissonND, dim=5).
Uses random collocation to avoid the curse of dimensionality.
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


def exact_solution(x):
    return jnp.sum(jnp.sin(jnp.pi / 2 * x))


def source_fn(x):
    return (jnp.pi**2) / 4 * jnp.sum(jnp.sin(jnp.pi / 2 * x))


def sample_boundary(key, n_per_face):
    """Sample n_per_face random points on each of the 2*DIM boundary faces."""
    keys = jax.random.split(key, 2 * DIM)
    all_pts = []
    for i in range(DIM):
        for side, val in enumerate([0.0, 1.0]):
            pts = jax.random.uniform(keys[2 * i + side], (n_per_face, DIM))
            pts = pts.at[:, i].set(val)
            all_pts.append(pts)
    return jnp.concatenate(all_pts, axis=0)


def run_experiment(M, Q_int, Q_bnd_per_face, R_m, seed=0):
    key = jax.random.PRNGKey(seed)
    k_params, k_int, k_bnd, k_eval = jax.random.split(key, 4)

    W, b = init_params_nd(k_params, M, R_m, DIM)

    x_int = jax.random.uniform(k_int, (Q_int, DIM))
    x_bnd = sample_boundary(k_bnd, Q_bnd_per_face)

    V_int, lap_V_int = compute_basis_nd(x_int, W, b, LO, HI)
    A_pde = -lap_V_int
    rhs_pde = jax.vmap(source_fn)(x_int)

    V_bnd = compute_basis_nd(x_bnd, W, b, LO, HI)[0]
    rhs_bnd = jax.vmap(exact_solution)(x_bnd)

    A = jnp.concatenate([A_pde, V_bnd], axis=0)
    rhs = jnp.concatenate([rhs_pde, rhs_bnd])

    beta, _, _, _ = scipy.linalg.lstsq(
        np.array(A), np.array(rhs), lapack_driver="gelsd"
    )

    x_eval = jax.random.uniform(k_eval, (10000, DIM))
    V_eval = compute_basis_nd(x_eval, W, b, LO, HI)[0]
    u_num = V_eval @ jnp.array(beta)
    u_exact = jax.vmap(exact_solution)(x_eval)

    err = jnp.abs(u_num - u_exact)
    return float(jnp.max(err)), float(jnp.sqrt(jnp.mean(err**2)))


def main():
    Q_INT = 10000
    Q_BND = 1000
    M_VALUES = [100, 200, 500, 1000, 2000, 3000]

    # R_m sweep first to find optimal
    R_M_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    SWEEP_M = 2000
    print("5D Poisson: -Δu = f on [0,1]^5")
    print(f"Q_int={Q_INT}, Q_bnd={Q_BND}/face ({2 * DIM * Q_BND} total)")

    print(f"\n--- R_m sweep (M={SWEEP_M}) ---")
    print(f"{'R_m':>6}  {'max_err':>12}  {'rms_err':>12}")
    print("-" * 36)
    rm_results = []
    for rm in R_M_VALUES:
        max_err, rms_err = run_experiment(SWEEP_M, Q_INT, Q_BND, rm)
        rm_results.append((rm, max_err, rms_err))
        print(f"{rm:>6.2f}  {max_err:>12.3e}  {rms_err:>12.3e}")

    best_rm = min(rm_results, key=lambda t: t[2])
    R_M = best_rm[0]
    print(f"\nBest R_m = {R_M} (RMSE={best_rm[2]:.3e})")

    # M scaling with best R_m
    print(f"\n--- M scaling (R_m={R_M}) ---")
    print(f"{'M':>6}  {'max_err':>12}  {'rms_err':>12}")
    print("-" * 36)

    results = []
    for M in M_VALUES:
        max_err, rms_err = run_experiment(M, Q_INT, Q_BND, R_M)
        results.append((M, max_err, rms_err))
        print(f"{M:>6}  {max_err:>12.3e}  {rms_err:>12.3e}")

    # Plot: M scaling
    from scipy.stats import linregress

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    M_arr = np.array([m for m, _, _ in results], dtype=float)
    R_arr = np.array([r for _, _, r in results], dtype=float)
    ax1.loglog(
        M_arr,
        R_arr,
        marker="o",
        color="C0",
        linestyle="-",
        linewidth=1.4,
        markersize=7,
    )
    slope, intercept, r_value, _, _ = linregress(np.log10(M_arr), np.log10(R_arr))
    M_fit = np.logspace(np.log10(M_arr.min()), np.log10(M_arr.max()), 200)
    ax1.loglog(
        M_fit,
        10 ** (slope * np.log10(M_fit) + intercept),
        linestyle="--",
        color="C0",
        linewidth=1.2,
        label=f"RMSE∝M$^{{{slope:.2f}}}$, R={r_value:.3f}",
    )
    ax1.set_xlabel("M (basis functions)", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title(f"5D Poisson — scaling ($R_m$={R_M})", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    # Plot: R_m sweep
    rm_arr = np.array([r for r, _, _ in rm_results], dtype=float)
    rms_arr = np.array([r for _, _, r in rm_results], dtype=float)
    ax2.semilogy(
        rm_arr,
        rms_arr,
        marker="s",
        color="C1",
        linestyle="-",
        linewidth=1.4,
        markersize=7,
    )
    ax2.set_xlabel("$R_m$", fontsize=12)
    ax2.set_ylabel("RMSE", fontsize=12)
    ax2.set_title(f"5D Poisson — $R_m$ sweep (M={SWEEP_M})", fontsize=11)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = Path(__file__).parent / "poisson5d.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
