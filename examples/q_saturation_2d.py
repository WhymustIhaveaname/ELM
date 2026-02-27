#!/usr/bin/env python3
"""
Q-saturation study for 2D Helmholtz across multiple NM values.

For each NM in {100, 400, 800, 1600} and each subdomain config (1×1, 2×2):
    M = NM // (Nx*Ny)
    Q_total swept; Qx = Qy = int(sqrt(Q_total / N)) per subdomain

Q_total values are chosen as 4k^2 so that Qx=Qy is integer for both configs.
Output: 1x4 subplot figure saved as q_saturation_2d.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
    NM_VALUES = [100, 400, 800, 1600]
    CONFIGS = [(1, 1), (2, 2)]
    Q_VALUES = [64, 144, 256, 400, 900, 1600, 2500, 3600]

    all_results = {}

    for NM in NM_VALUES:
        print(f"\n===== NM={NM} =====")
        print(f"{'config':>7}  {'M':>6}  {'Q_total':>8}  {'Qs':>4}  {'RMSE':>12}")
        print("-" * 50)
        for Nx, Ny in CONFIGS:
            N = Nx * Ny
            M = NM // N
            pts = []
            for Q in Q_VALUES:
                Qs = int(round((Q / N) ** 0.5))
                rms = run_experiment(NM, Nx, Ny, Q)
                pts.append((Q, rms))
                print(f"  {Nx}×{Ny}  {M:>6}  {Q:>8}  {Qs:>4}  {rms:>12.3e}")
            all_results[(NM, Nx, Ny)] = pts
            print()

    style = {
        (1, 1): dict(color="C0", marker="o"),
        (2, 2): dict(color="C2", marker="^"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    for idx, NM in enumerate(NM_VALUES):
        ax = axes[idx]
        for Nx, Ny in CONFIGS:
            M = NM // (Nx * Ny)
            pts = all_results[(NM, Nx, Ny)]
            Q_arr = np.array([q for q, _ in pts], dtype=float)
            R_arr = np.array([r for _, r in pts], dtype=float)

            st = style[(Nx, Ny)]
            ax.loglog(
                Q_arr,
                R_arr,
                marker=st["marker"],
                color=st["color"],
                linestyle="-",
                linewidth=1.4,
                markersize=6,
                label=f"{Nx}×{Ny} (M={M})",
            )

        ax.axvline(x=NM, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("$Q_{total}$", fontsize=11)
        ax.set_title(f"NM = {NM}", fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("RMSE", fontsize=11)
    fig.suptitle(
        f"Q-saturation — 2D Helmholtz  ($R_m$={R_M}, seed={SEED})",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / "q_saturation_2d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
