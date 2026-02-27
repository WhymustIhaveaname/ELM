#!/usr/bin/env python3
"""
Q-saturation study across multiple NM values.

For each NM in {128, 256, 512, 1024} and each N in {1, 2, 4}:
    M = NM // N
    Q_total swept over Q_VALUES  (total collocation points across all subdomains)
    Q_per_subdomain = Q_total // N

Output: 1x4 subplot figure, each panel for one NM value, three lines per panel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from helm1d import DOMAIN, LAMBDA, exact_solution, source_fn
from locelm import evaluate_solution, solve_locelm_1d

jax.config.update("jax_enable_x64", True)

R_M = 3.0
SEED = 0
N_EVAL = 1000


def run_experiment(NM, N, Q_total):
    M = NM // N
    bc_left = float(exact_solution(jnp.array(DOMAIN[0])))
    bc_right = float(exact_solution(jnp.array(DOMAIN[1])))

    beta, subdomains = solve_locelm_1d(
        pde_coeffs=(1.0, 0.0, -LAMBDA),
        source_fn=source_fn,
        bc_left=bc_left,
        bc_right=bc_right,
        domain=DOMAIN,
        N_e=N,
        Q=Q_total // N,
        M=M,
        R_m=R_M,
        seed=SEED,
    )

    x_eval = jnp.linspace(DOMAIN[0], DOMAIN[1], N_EVAL)
    u_num = evaluate_solution(x_eval, beta, subdomains, M)
    u_true = jax.vmap(exact_solution)(x_eval)

    return float(jnp.sqrt(jnp.mean((u_num - u_true) ** 2)))


def main():
    NM_VALUES = [128, 256, 512, 1024]
    N_VALUES = [1, 2, 4]
    Q_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    all_results = {}

    for NM in NM_VALUES:
        print(f"\n===== NM={NM} =====")
        print(f"{'N':>4}  {'M':>6}  {'Q_total':>8}  {'Q/sub':>6}  {'RMSE':>12}")
        print("-" * 48)
        for N in N_VALUES:
            M = NM // N
            pts = []
            for Q in Q_VALUES:
                rms = run_experiment(NM, N, Q)
                pts.append((Q, rms))
                print(f"{N:>4}  {M:>6}  {Q:>8}  {Q // N:>6}  {rms:>12.3e}")
            all_results[(NM, N)] = pts
            print()

    style = {
        1: dict(color="C0", marker="o"),
        2: dict(color="C1", marker="s"),
        4: dict(color="C2", marker="^"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    for idx, NM in enumerate(NM_VALUES):
        ax = axes[idx]
        for N in N_VALUES:
            M = NM // N
            pts = all_results[(NM, N)]
            Q_arr = np.array([q for q, _ in pts], dtype=float)
            R_arr = np.array([r for _, r in pts], dtype=float)

            st = style[N]
            ax.loglog(
                Q_arr,
                R_arr,
                marker=st["marker"],
                color=st["color"],
                linestyle="-",
                linewidth=1.4,
                markersize=6,
                label=f"N={N} (M={M})",
            )

        ax.axvline(x=NM, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("$Q_{total}$", fontsize=11)
        ax.set_title(f"NM = {NM}", fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("RMSE", fontsize=11)
    fig.suptitle(
        f"Q-saturation — 1D Helmholtz  ($R_m$={R_M}, seed={SEED})",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / "q_saturation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
