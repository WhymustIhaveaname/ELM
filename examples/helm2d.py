"""
Reproduce 2D Helmholtz equation test from locELM paper (arXiv:2012.02895).

Problem (Helm2d.tex lines 8-36):
    d²u/dx² + d²u/dy² - 10u = f(x,y),  (x,y) in [0, 3.6]^2
    u = -[3/2 cos(πx+2π/5) + 2cos(2πx-π/5)] * [3/2 cos(πy+2π/5) + 2cos(2πy-π/5)]

Paper results (Helm2d.tex Table 1, lines 311-328):
    locELM 2x2: Q=25x25, M=400, R_m=1.5 -> max_err=2.01e-5, rms_err=1.41e-6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from locelm import evaluate_solution_2d, solve_locelm_2d

jax.config.update("jax_enable_x64", True)

LAMBDA = 10.0


def _phi_x(x):
    return -(
        1.5 * jnp.cos(jnp.pi * x + 2 * jnp.pi / 5)
        + 2.0 * jnp.cos(2 * jnp.pi * x - jnp.pi / 5)
    )


def _phi_xx(x):
    """Second derivative of _phi_x."""
    return 1.5 * jnp.pi**2 * jnp.cos(jnp.pi * x + 2 * jnp.pi / 5) + 2.0 * (
        2 * jnp.pi
    ) ** 2 * jnp.cos(2 * jnp.pi * x - jnp.pi / 5)


def exact_solution(xy):
    return _phi_x(xy[0]) * _phi_x(xy[1])


def source_fn(xy):
    """f = d²u/dx² + d²u/dy² - λu, computed from the separable structure."""
    x, y = xy[0], xy[1]
    laplacian = _phi_xx(x) * _phi_x(y) + _phi_x(x) * _phi_xx(y)
    return laplacian - LAMBDA * _phi_x(x) * _phi_x(y)


def run(Nx, Ny, Qx, Qy, M, R_m=1.5, seed=2, n_eval=50):
    domain = (0.0, 3.6, 0.0, 3.6)
    pde_coeffs = (1.0, 1.0, -LAMBDA)

    beta, subdomains = solve_locelm_2d(
        pde_coeffs,
        source_fn,
        exact_solution,
        domain,
        Nx,
        Ny,
        Qx,
        Qy,
        M,
        R_m,
        seed,
    )

    xx, yy = jnp.meshgrid(
        jnp.linspace(0.0, 3.6, n_eval),
        jnp.linspace(0.0, 3.6, n_eval),
        indexing="ij",
    )
    xy_eval = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    u_num = evaluate_solution_2d(xy_eval, beta, subdomains, Nx, Ny, M)
    u_exact = jax.vmap(exact_solution)(xy_eval)

    err = jnp.abs(u_num - u_exact)
    return float(jnp.max(err)), float(jnp.sqrt(jnp.mean(err**2)))


def main():
    print("=" * 60)
    print("2D Helmholtz: Laplacian(u) - 10u = f, [0,3.6]^2")
    print("=" * 60)

    print("\n--- locELM 2x2, Q=25x25, M=400, R_m=1.5 ---")
    print("Paper Table 1: max=2.01e-5, rms=1.41e-6")
    max_err, rms_err = run(2, 2, 25, 25, 400, R_m=1.5)
    print(f"  max error = {max_err:.3e}")
    print(f"  rms error = {rms_err:.3e}")

    print("\n--- Parameter comparison (Table 2) ---")
    print(
        f"{'Q':>7s}  {'M':>5s}  {'max_err':>12s}  {'rms_err':>12s}  {'paper_max':>12s}  {'paper_rms':>12s}"
    )
    configs = [
        (20, 20, 300, 7.28e-4, 5.28e-5),
        (25, 25, 400, 2.01e-5, 1.41e-6),
    ]
    for qx, qy, m_val, p_max, p_rms in configs:
        max_err, rms_err = run(2, 2, qx, qy, m_val, R_m=1.5)
        print(
            f"{qx}x{qy:>2d}  {m_val:>5d}  {max_err:>12.3e}  {rms_err:>12.3e}  {p_max:>12.3e}  {p_rms:>12.3e}"
        )


if __name__ == "__main__":
    main()
