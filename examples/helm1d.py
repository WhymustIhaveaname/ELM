"""
Reproduce 1D Helmholtz equation test from locELM paper (arXiv:2012.02895).

Problem (Helm1d.tex lines 2-44):
    u''(x) - 10*u(x) = f(x),  x in [0, 8]
    u(x) = sin(3*pi*x + 3*pi/20) * cos(2*pi*x + pi/10) + 2  (exact)
    Dirichlet BCs from exact solution.

Paper results (Helm1d.tex Table 1, lines 588-607):
    locELM: N_e=4, Q=100, M=100, R_m=3.0 -> max_err=1.56e-9, rms_err=2.25e-10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from locelm import solve_locelm_1d, evaluate_solution

jax.config.update("jax_enable_x64", True)

LAMBDA = 10.0
DOMAIN = (0.0, 8.0)


def exact_solution(x):
    return (
        jnp.sin(3 * jnp.pi * x + 3 * jnp.pi / 20)
        * jnp.cos(2 * jnp.pi * x + jnp.pi / 10)
        + 2.0
    )


def exact_solution_d2(x):
    a1 = 3 * jnp.pi
    phi1 = 3 * jnp.pi / 20
    a2 = 2 * jnp.pi
    phi2 = jnp.pi / 10
    s1 = jnp.sin(a1 * x + phi1)
    c1 = jnp.cos(a1 * x + phi1)
    s2 = jnp.sin(a2 * x + phi2)
    c2 = jnp.cos(a2 * x + phi2)
    return -(a1**2 + a2**2) * s1 * c2 - 2 * a1 * a2 * c1 * s2


def source_fn(x):
    return exact_solution_d2(x) - LAMBDA * exact_solution(x)


def run(N_e, Q, M, R_m=3.0, seed=0, n_eval=1000):
    bc_left = float(exact_solution(jnp.array(DOMAIN[0])))
    bc_right = float(exact_solution(jnp.array(DOMAIN[1])))

    beta, subdomains = solve_locelm_1d(
        pde_coeffs=(1.0, 0.0, -LAMBDA),
        source_fn=source_fn,
        bc_left=bc_left,
        bc_right=bc_right,
        domain=DOMAIN,
        N_e=N_e,
        Q=Q,
        M=M,
        R_m=R_m,
        seed=seed,
    )

    x_eval = jnp.linspace(DOMAIN[0], DOMAIN[1], n_eval)
    u_num = evaluate_solution(x_eval, beta, subdomains, M)
    u_exact = jax.vmap(exact_solution)(x_eval)

    err = jnp.abs(u_num - u_exact)
    max_err = float(jnp.max(err))
    rms_err = float(jnp.sqrt(jnp.mean(err**2)))

    return max_err, rms_err, x_eval, u_num, u_exact, err


def main():
    print("=" * 60)
    print("1D Helmholtz: u'' - 10u = f,  x in [0, 8]")
    print("=" * 60)

    # Step 1: Global ELM (N_e=1)
    print("\n--- Global ELM (N_e=1, Q=50, M=50, R_m=3.0) ---")
    print("Paper: max error ~10^1 (single subdomain, poor accuracy expected)")
    max_err, rms_err, *_ = run(N_e=1, Q=50, M=50, R_m=3.0)
    print(f"  max error = {max_err:.3e}")
    print(f"  rms error = {rms_err:.3e}")

    # Step 2: locELM (N_e=4) — Table 1 target
    print("\n--- locELM (N_e=4, Q=100, M=100, R_m=3.0) ---")
    print("Paper Table 1: max=1.56e-9, rms=2.25e-10")
    max_err, rms_err, *_ = run(N_e=4, Q=100, M=100, R_m=3.0)
    print(f"  max error = {max_err:.3e}")
    print(f"  rms error = {rms_err:.3e}")

    # Step 3: Parameter sweep — Table 2
    print("\n--- Parameter sweep (N_e=4, Q=100, R_m=3.0) ---")
    print(
        f"{'M':>5s}  {'max_err':>12s}  {'rms_err':>12s}  {'paper_max':>12s}  {'paper_rms':>12s}"
    )
    paper_results = {
        75: (4.02e-8, 5.71e-9),
        100: (1.56e-9, 2.25e-10),
        125: (1.42e-10, 2.55e-11),
    }
    for M_val in [75, 100, 125]:
        max_err, rms_err, *_ = run(N_e=4, Q=100, M=M_val, R_m=3.0)
        p_max, p_rms = paper_results[M_val]
        print(
            f"{M_val:>5d}  {max_err:>12.3e}  {rms_err:>12.3e}  {p_max:>12.3e}  {p_rms:>12.3e}"
        )


if __name__ == "__main__":
    main()
