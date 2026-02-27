#!/usr/bin/env python3
"""
Scaling law study: RMSE vs P (total parameters) for 2D Helmholtz.

P = 4 * NM  (4 params per 2D neuron: W_x, W_y, bias, output coeff)
Configs: 1×1 (vanilla), 2×2 (paper)
NM in {100, 200, 400, 800, 1600}  =>  M = NM // N

Fixed: Q_total = 2500 (= 4×25², giving Qx=Qy=50 for 1×1, 25 for 2×2).
R_m=1.5, domain=[0,3.6]^2, seed=2.

Output: log-log plot of RMSE vs P, dashed best-fit lines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from helm2d import DOMAIN, LAMBDA, exact_solution, source_fn
from locelm import evaluate_solution_2d, solve_locelm_2d

jax.config.update("jax_enable_x64", True)

R_M = 1.5
SEED = 2
N_EVAL = 200


def run_experiment(NM, Nx, Ny, Q_total):
    N = Nx * Ny
    M = NM // N
    Qs = int(round((Q_total / N) ** 0.5))

    beta, subdomains = solve_locelm_2d(
        pde_coeffs=(1.0, 1.0, -LAMBDA),
        source_fn=source_fn,
        bc_fn=exact_solution,
        domain=DOMAIN,
        Nx=Nx,
        Ny=Ny,
        Qx=Qs,
        Qy=Qs,
        M=M,
        R_m=R_M,
        seed=SEED,
    )

    xx, yy = jnp.meshgrid(
        jnp.linspace(DOMAIN[0], DOMAIN[1], N_EVAL),
        jnp.linspace(DOMAIN[2], DOMAIN[3], N_EVAL),
        indexing="ij",
    )
    xy_eval = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    u_num = evaluate_solution_2d(xy_eval, beta, subdomains, Nx, Ny, M)
    u_exact = jax.vmap(exact_solution)(xy_eval)

    return float(jnp.sqrt(jnp.mean((u_num - u_exact) ** 2)))


def main():
    CONFIGS = [(1, 1), (2, 2)]
    NM_VALUES = [100, 200, 400, 800, 1600]
    Q_TOTAL = 2500

    print(f"Q_total={Q_TOTAL}, R_m={R_M}, domain={DOMAIN}, seed={SEED}")
    print(
        f"\n{'config':>7}  {'NM':>6}  {'M':>6}  {'Qs':>4}  {'P=4NM':>8}  {'RMSE':>12}"
    )
    print("-" * 56)

    results = {}
    for Nx, Ny in CONFIGS:
        N = Nx * Ny
        Qs = int(round((Q_TOTAL / N) ** 0.5))
        pts = []
        for NM in NM_VALUES:
            M = NM // N
            P = 4 * NM
            rms = run_experiment(NM, Nx, Ny, Q_TOTAL)
            pts.append((P, rms))
            print(f"  {Nx}×{Ny}  {NM:>6}  {M:>6}  {Qs:>4}  {P:>8}  {rms:>12.3e}")
        results[(Nx, Ny)] = pts

    print()
    print(f"{'config':>7}  {'slope':>14}  {'intercept':>12}  {'R':>8}  {'R^2':>8}")
    print("-" * 56)

    fig, ax = plt.subplots(figsize=(7, 5))

    style = {
        (1, 1): dict(color="C0", marker="o", label_prefix="1×1 (vanilla)"),
        (2, 2): dict(color="C2", marker="^", label_prefix="2×2 (paper)"),
    }

    for Nx, Ny in CONFIGS:
        P_arr = np.array([p for p, _ in results[(Nx, Ny)]], dtype=float)
        R_arr = np.array([r for _, r in results[(Nx, Ny)]], dtype=float)

        slope, intercept, r_value, _, _ = linregress(np.log10(P_arr), np.log10(R_arr))

        print(
            f"  {Nx}×{Ny}  {slope:>14.4f}  {intercept:>12.4f}  "
            f"{r_value:>8.4f}  {r_value**2:>8.4f}"
        )

        st = style[(Nx, Ny)]
        ax.loglog(
            P_arr,
            R_arr,
            marker=st["marker"],
            color=st["color"],
            linestyle="-",
            linewidth=1.4,
            markersize=7,
            label=st["label_prefix"],
        )

        P_fit = np.logspace(np.log10(P_arr.min()), np.log10(P_arr.max()), 200)
        ax.loglog(
            P_fit,
            10 ** (slope * np.log10(P_fit) + intercept),
            linestyle="--",
            color=st["color"],
            linewidth=1.2,
            label=f"fit {Nx}×{Ny}: RMSE∝P$^{{{slope:.2f}}}$, R={r_value:.3f}",
        )

    ax.set_xlabel("P  (total parameters = 4NM)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(
        f"Scaling law — 2D Helmholtz  ($Q_{{total}}$={Q_TOTAL}, $R_m$={R_M})",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = Path(__file__).parent / "scaling_law_2d.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
