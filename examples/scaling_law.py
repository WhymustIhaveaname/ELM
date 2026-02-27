#!/usr/bin/env python3
"""
Scaling law study: RMSE vs P (total parameters) for 1D Helmholtz.

P = 3 * N * M  (3 params per neuron: input weight w, bias b, output coeff beta)
N in {1 (vanilla), 2, 4 (paper)}
NM in {128, 256, 512, 1024}  =>  M = NM // N  =>  P = 3 * NM (independent of N)

Fixed: Q_total = max(NM_VALUES) total collocation points, R_m=3.0, domain=[0,8].
Q_total >= NM guarantees the system is always overdetermined.

Output: log-log plot of RMSE vs P with one line per N value,
        dashed best-fit lines, and printed scaling law coefficients.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from helm1d import DOMAIN, LAMBDA, exact_solution, source_fn
from locelm import evaluate_solution, solve_locelm_1d

jax.config.update("jax_enable_x64", True)

R_M = 3.0
SEED = 0
N_EVAL = 1000


def run_experiment(N, M, Q_total):
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
    N_VALUES = [1, 2, 4]
    NM_VALUES = [128, 256, 512, 1024]
    Q_TOTAL = max(NM_VALUES)

    print(f"Q_total={Q_TOTAL}, R_m={R_M}, domain={DOMAIN}, seed={SEED}")
    print(f"\n{'N':>4}  {'NM':>6}  {'M':>6}  {'Q/sub':>6}  {'P=3NM':>8}  {'RMSE':>12}")
    print("-" * 54)

    results = {}
    for N in N_VALUES:
        pts = []
        for NM in NM_VALUES:
            M = NM // N
            P = 3 * NM
            rms = run_experiment(N, M, Q_TOTAL)
            pts.append((P, rms))
            print(f"{N:>4}  {NM:>6}  {M:>6}  {Q_TOTAL // N:>6}  {P:>8}  {rms:>12.3e}")
        results[N] = pts

    print()
    print(f"{'N':>4}  {'slope (alpha)':>14}  {'intercept':>12}  {'R':>8}  {'R^2':>8}")
    print("-" * 56)

    fig, ax = plt.subplots(figsize=(7, 5))

    style = {
        1: dict(color="C0", marker="o", label_prefix="N=1 (vanilla)"),
        2: dict(color="C1", marker="s", label_prefix="N=2"),
        4: dict(color="C2", marker="^", label_prefix="N=4 (paper)"),
    }

    for N in N_VALUES:
        P_arr = np.array([p for p, _ in results[N]], dtype=float)
        R_arr = np.array([r for _, r in results[N]], dtype=float)

        slope, intercept, r_value, _, _ = linregress(np.log10(P_arr), np.log10(R_arr))

        print(
            f"{N:>4}  {slope:>14.4f}  {intercept:>12.4f}  "
            f"{r_value:>8.4f}  {r_value**2:>8.4f}"
        )

        st = style[N]
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
            label=f"fit N={N}: RMSE∝P$^{{{slope:.2f}}}$, R={r_value:.3f}",
        )

    ax.set_xlabel("P  (total parameters = 3NM)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(
        f"Scaling law — 1D Helmholtz  ($Q_{{total}}$={Q_TOTAL}, $R_m$={R_M})",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = Path(__file__).parent / "scaling_law.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
