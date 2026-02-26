"""
locELM: Local Extreme Learning Machines for solving 1D linear PDEs.

Reproduces: S. Dong & Z. Li, "Local Extreme Learning Machines and Domain
Decomposition for Solving Linear and Nonlinear PDEs", CMAME 2021.
(arXiv:2012.02895)
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def init_params(key, M, R_m):
    """Initialize random hidden-layer weights and biases for one local ELM."""
    k1, k2 = jax.random.split(key)
    w = jax.random.uniform(k1, (M,), minval=-R_m, maxval=R_m)
    b = jax.random.uniform(k2, (M,), minval=-R_m, maxval=R_m)
    return w, b


def compute_basis(x_points, w, b, x_lo, x_hi):
    """Compute V_j, dV_j/dx, d²V_j/dx² at collocation points.

    Uses analytical derivatives of tanh: tanh'(z) = 1 - tanh²(z).

    Args:
        x_points: (Q,) collocation points in [x_lo, x_hi]
        w, b: (M,) hidden layer params
        x_lo, x_hi: subdomain bounds

    Returns:
        V, dV, d2V: each (Q, M)
    """
    scale = 2.0 / (x_hi - x_lo)
    x_norm = 2.0 * (x_points[:, None] - x_lo) / (x_hi - x_lo) - 1.0
    z = w[None, :] * x_norm + b[None, :]

    V = jnp.tanh(z)
    sech2 = 1.0 - V**2
    dV = sech2 * w[None, :] * scale
    d2V = -2.0 * V * sech2 * (w[None, :] * scale) ** 2

    return V, dV, d2V


def compute_basis_at_point(x, w, b, x_lo, x_hi):
    """Compute V_j and dV_j/dx at a single point. Returns two (M,) arrays."""
    scale = 2.0 / (x_hi - x_lo)
    x_norm = 2.0 * (x - x_lo) / (x_hi - x_lo) - 1.0
    z = w * x_norm + b

    V = jnp.tanh(z)
    sech2 = 1.0 - V**2
    dV = sech2 * w * scale

    return V, dV


def solve_locelm_1d(
    pde_coeffs, source_fn, bc_left, bc_right, domain, N_e, Q, M, R_m, seed=1
):
    """Solve a 1D linear PDE with Dirichlet BCs using locELM.

    PDE: a2 * u'' + a1 * u' + a0 * u = f(x) on [domain[0], domain[1]]

    For 2nd-order PDE, C^1 continuity is imposed across subdomain boundaries.

    Returns:
        beta: (N_e * M,) least-squares solution coefficients
        subdomains: list of dicts with network params and collocation data
    """
    a, b = domain
    a2, a1, a0 = pde_coeffs
    boundaries = jnp.linspace(a, b, N_e + 1)

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, N_e)

    subdomains = []
    for s in range(N_e):
        x_lo, x_hi = float(boundaries[s]), float(boundaries[s + 1])
        ws, bs = init_params(keys[s], M, R_m)
        x_pts = jnp.linspace(x_lo, x_hi, Q)
        V, dV, d2V = compute_basis(x_pts, ws, bs, x_lo, x_hi)
        LV = a2 * d2V + a1 * dV + a0 * V
        subdomains.append(
            {
                "w": ws,
                "b": bs,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "x_points": x_pts,
                "V": V,
                "dV": dV,
                "d2V": d2V,
                "LV": LV,
            }
        )

    total_unknowns = N_e * M

    # --- Assemble linear system A @ beta = rhs ---

    # PDE residual: N_e * Q rows
    n_pde = N_e * Q
    A_pde = jnp.zeros((n_pde, total_unknowns))
    rhs_pde = jnp.zeros(n_pde)
    for s in range(N_e):
        sd = subdomains[s]
        A_pde = A_pde.at[s * Q : (s + 1) * Q, s * M : (s + 1) * M].set(sd["LV"])
        f_vals = jax.vmap(source_fn)(sd["x_points"])
        rhs_pde = rhs_pde.at[s * Q : (s + 1) * Q].set(f_vals)

    # Boundary conditions: 2 rows
    A_bc = jnp.zeros((2, total_unknowns))
    rhs_bc = jnp.array([bc_left, bc_right])

    V_left, _ = compute_basis_at_point(
        a,
        subdomains[0]["w"],
        subdomains[0]["b"],
        subdomains[0]["x_lo"],
        subdomains[0]["x_hi"],
    )
    A_bc = A_bc.at[0, :M].set(V_left)

    V_right, _ = compute_basis_at_point(
        b,
        subdomains[-1]["w"],
        subdomains[-1]["b"],
        subdomains[-1]["x_lo"],
        subdomains[-1]["x_hi"],
    )
    A_bc = A_bc.at[1, (N_e - 1) * M : N_e * M].set(V_right)

    # C^1 continuity: 2 * (N_e - 1) rows
    n_cont = 2 * (N_e - 1)
    A_cont = jnp.zeros((n_cont, total_unknowns))
    rhs_cont = jnp.zeros(n_cont)

    for s in range(N_e - 1):
        xb = float(boundaries[s + 1])
        sd_l, sd_r = subdomains[s], subdomains[s + 1]

        V_l, dV_l = compute_basis_at_point(
            xb, sd_l["w"], sd_l["b"], sd_l["x_lo"], sd_l["x_hi"]
        )
        V_r, dV_r = compute_basis_at_point(
            xb, sd_r["w"], sd_r["b"], sd_r["x_lo"], sd_r["x_hi"]
        )

        row_c0 = 2 * s
        A_cont = A_cont.at[row_c0, s * M : (s + 1) * M].set(V_l)
        A_cont = A_cont.at[row_c0, (s + 1) * M : (s + 2) * M].set(-V_r)

        row_c1 = 2 * s + 1
        A_cont = A_cont.at[row_c1, s * M : (s + 1) * M].set(dV_l)
        A_cont = A_cont.at[row_c1, (s + 1) * M : (s + 2) * M].set(-dV_r)

    A = jnp.concatenate([A_pde, A_bc, A_cont], axis=0)
    rhs = jnp.concatenate([rhs_pde, rhs_bc, rhs_cont])

    beta, _, _, _ = scipy.linalg.lstsq(
        np.array(A), np.array(rhs), lapack_driver="gelsd"
    )
    beta = jnp.array(beta)

    return beta, subdomains


def evaluate_solution(x_eval, beta, subdomains, M):
    """Evaluate the locELM solution at arbitrary points."""
    N_e = len(subdomains)
    u_eval = jnp.zeros_like(x_eval)

    for s in range(N_e):
        sd = subdomains[s]
        if s < N_e - 1:
            mask = (x_eval >= sd["x_lo"]) & (x_eval < sd["x_hi"])
        else:
            mask = (x_eval >= sd["x_lo"]) & (x_eval <= sd["x_hi"])

        x_norm = 2.0 * (x_eval - sd["x_lo"]) / (sd["x_hi"] - sd["x_lo"]) - 1.0
        z = sd["w"][None, :] * x_norm[:, None] + sd["b"][None, :]
        V = jnp.tanh(z)

        beta_s = beta[s * M : (s + 1) * M]
        u_s = V @ beta_s
        u_eval = u_eval + u_s * mask

    return u_eval
