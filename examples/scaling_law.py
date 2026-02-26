#!/usr/bin/env python3
"""
Scaling law study: RMSE vs P (total parameters) for 1D Helmholtz.

P = 3 * N * M  (3 params per neuron: input weight w, bias b, output coeff beta)
N in {1 (vanilla), 2, 4 (paper)}
NM in {128, 256, 512, 1024}  =>  M = NM // N  =>  P = 3 * NM (independent of N)

Fixed: Q=100 collocation points per subdomain, R_m=3.0, domain=[0,8].
Note: for N=1 with large M (>100), the system is underdetermined (Q < M);
lstsq returns the minimum-norm solution.

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

from locelm import evaluate_solution, solve_locelm_1d

jax.config.update("jax_enable_x64", True)

# ---------- Problem: 1D Helmholtz  u'' - 10u = f  on [0, 8] ----------------
LAMBDA = 10.0
DOMAIN = (0.0, 8.0)
R_M = 3.0
Q = 100       # collocation points per subdomain (fixed throughout)
SEED = 0
N_EVAL = 1000


def exact_solution(x):
    return (
        jnp.sin(3 * jnp.pi * x + 3 * jnp.pi / 20)
        * jnp.cos(2 * jnp.pi * x + jnp.pi / 10)
        + 2.0
    )


def exact_solution_d2(x):
    """Analytical second derivative of the exact solution."""
    a1, phi1 = 3 * jnp.pi, 3 * jnp.pi / 20
    a2, phi2 = 2 * jnp.pi, jnp.pi / 10
    s1 = jnp.sin(a1 * x + phi1)
    c1 = jnp.cos(a1 * x + phi1)
    s2 = jnp.sin(a2 * x + phi2)
    c2 = jnp.cos(a2 * x + phi2)
    return -(a1**2 + a2**2) * s1 * c2 - 2 * a1 * a2 * c1 * s2


def source_fn(x):
    return exact_solution_d2(x) - LAMBDA * exact_solution(x)


# ---------- Experiment runner -----------------------------------------------

def run_experiment(N, M):
    """Run locELM for N subdomains and M basis functions per subdomain.

    Returns RMSE on 1000 uniformly-spaced evaluation points.
    """
    bc_left = float(exact_solution(jnp.array(DOMAIN[0])))
    bc_right = float(exact_solution(jnp.array(DOMAIN[1])))

    beta, subdomains = solve_locelm_1d(
        pde_coeffs=(1.0, 0.0, -LAMBDA),
        source_fn=source_fn,
        bc_left=bc_left,
        bc_right=bc_right,
        domain=DOMAIN,
        N_e=N,
        Q=Q,
        M=M,
        R_m=R_M,
        seed=SEED,
    )

    x_eval = jnp.linspace(DOMAIN[0], DOMAIN[1], N_EVAL)
    u_num = evaluate_solution(x_eval, beta, subdomains, M)
    u_true = jax.vmap(exact_solution)(x_eval)

    return float(jnp.sqrt(jnp.mean((u_num - u_true) ** 2)))


# ---------- Main ------------------------------------------------------------

def main():
    N_VALUES = [1, 2, 4]
    NM_VALUES = [128, 256, 512, 1024]

    style = {
        1: dict(color="C0", marker="o", label_prefix="N=1 (vanilla)"),
        2: dict(color="C1", marker="s", label_prefix="N=2"),
        4: dict(color="C2", marker="^", label_prefix="N=4 (paper)"),
    }

    print(f"Q={Q}, R_m={R_M}, domain={DOMAIN}, seed={SEED}")
    print(f"\n{'N':>4}  {'NM':>6}  {'M':>6}  {'P=3NM':>8}  {'RMSE':>12}")
    print("-" * 46)

    results = {}
    for N in N_VALUES:
        pts = []
        for NM in NM_VALUES:
            M = NM // N
            P = 3 * NM          # = 3 * N * M, independent of N
            rms = run_experiment(N, M)
            pts.append((P, rms))
            print(f"{N:>4}  {NM:>6}  {M:>6}  {P:>8}  {rms:>12.3e}")
        results[N] = pts

    # ---------- Fit and print -----------------------------------------------
    print()
    print(f"{'N':>4}  {'slope (alpha)':>14}  {'intercept':>12}  {'R':>8}  {'R^2':>8}")
    print("-" * 56)

    fig, ax = plt.subplots(figsize=(7, 5))

    for N in N_VALUES:
        P_arr = np.array([p for p, _ in results[N]], dtype=float)
        R_arr = np.array([r for _, r in results[N]], dtype=float)

        log_P = np.log10(P_arr)
        log_R = np.log10(R_arr)
        slope, intercept, r_value, _, _ = linregress(log_P, log_R)
        r2 = r_value**2

        print(
            f"{N:>4}  {slope:>14.4f}  {intercept:>12.4f}  "
            f"{r_value:>8.4f}  {r2:>8.4f}"
        )

        st = style[N]
        fit_label = (
            f"fit N={N}: RMSE∝P$^{{{slope:.2f}}}$, R={r_value:.3f}"
        )

        # Data points
        ax.loglog(
            P_arr, R_arr,
            marker=st["marker"], color=st["color"],
            linestyle="-", linewidth=1.4, markersize=7,
            label=st["label_prefix"],
        )

        # Dashed fit line spanning data range
        P_fit = np.logspace(np.log10(P_arr.min()), np.log10(P_arr.max()), 200)
        R_fit = 10 ** (slope * np.log10(P_fit) + intercept)
        ax.loglog(
            P_fit, R_fit,
            linestyle="--", color=st["color"], linewidth=1.2,
            label=fit_label,
        )

    ax.set_xlabel("P  (total parameters = 3NM)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(
        f"Scaling law — 1D Helmholtz  (Q={Q}, $R_m$={R_M})",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = Path(__file__).parent / "scaling_law.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
