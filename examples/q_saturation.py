#!/usr/bin/env python3
"""
Q-saturation study: fix NM=1024, vary Q to find the saturation point.

For each N in {1, 2, 4}:
    M = 1024 // N
    Q swept over Q_VALUES

The system transitions from underdetermined (Q < M) to overdetermined (Q > M)
per subdomain. RMSE should plateau once Q is large enough.

Output: log-log plot of RMSE vs Q with one line per N value.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from locelm import evaluate_solution, solve_locelm_1d

jax.config.update("jax_enable_x64", True)

# ---------- Problem: 1D Helmholtz  u'' - 10u = f  on [0, 8] ----------------
LAMBDA = 10.0
DOMAIN = (0.0, 8.0)
R_M = 3.0
SEED = 0
N_EVAL = 1000
NM = 1024   # fixed: N * M = 1024


def exact_solution(x):
    return (
        jnp.sin(3 * jnp.pi * x + 3 * jnp.pi / 20)
        * jnp.cos(2 * jnp.pi * x + jnp.pi / 10)
        + 2.0
    )


def exact_solution_d2(x):
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

def run_experiment(N, M, Q):
    """Run locELM for N subdomains, M basis functions, Q collocation pts/subdomain."""
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
    Q_VALUES = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]

    style = {
        1: dict(color="C0", marker="o", label="N=1 (vanilla)"),
        2: dict(color="C1", marker="s", label="N=2"),
        4: dict(color="C2", marker="^", label="N=4 (paper)"),
    }

    print(f"NM={NM}, R_m={R_M}, domain={DOMAIN}, seed={SEED}")
    print(f"\n{'N':>4}  {'M':>6}  {'Q':>6}  {'Q/M':>6}  {'RMSE':>12}")
    print("-" * 44)

    results = {}   # N -> list of (Q, rmse)
    for N in N_VALUES:
        M = NM // N
        pts = []
        for Q in Q_VALUES:
            rms = run_experiment(N, M, Q)
            pts.append((Q, rms))
            ratio = Q / M
            marker = "<" if Q < M else ("=" if Q == M else ">")
            print(f"{N:>4}  {M:>6}  {Q:>6}  {ratio:>5.2f}{marker}  {rms:>12.3e}")
        results[N] = pts
        print()

    # ---------- Plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    for N in N_VALUES:
        M = NM // N
        Q_arr = np.array([q for q, _ in results[N]], dtype=float)
        R_arr = np.array([r for _, r in results[N]], dtype=float)

        st = style[N]
        ax.loglog(
            Q_arr, R_arr,
            marker=st["marker"], color=st["color"],
            linestyle="-", linewidth=1.4, markersize=7,
            label=f"{st['label']}  (M={M})",
        )

        # Vertical dashed line at Q = M (transition point)
        ax.axvline(x=M, color=st["color"], linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Q  (collocation points per subdomain)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(
        f"Q-saturation — 1D Helmholtz  (NM={NM}, $R_m$={R_M})",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Annotate transition lines
    ax.text(
        1024 * 1.05, ax.get_ylim()[0] * 10,
        "Q=M\n(N=1)", fontsize=7, color="C0", alpha=0.7,
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / "q_saturation.png"
    plt.savefig(out_path, dpi=150)
    print(f"Figure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
