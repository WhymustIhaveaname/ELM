"""
locELM: Local Extreme Learning Machines for solving 1D/2D linear PDEs.

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


# ===== 2D functions =====


def init_params_2d(key, M, R_m):
    """Initialize random hidden-layer weights and biases for one 2D local ELM."""
    k1, k2 = jax.random.split(key)
    W = jax.random.uniform(k1, (M, 2), minval=-R_m, maxval=R_m)
    b = jax.random.uniform(k2, (M,), minval=-R_m, maxval=R_m)
    return W, b


def compute_basis_2d(xy_points, W, b, x_lo, x_hi, y_lo, y_hi):
    """Compute V, dV/dx, dV/dy, d²V/dx², d²V/dy² at 2D collocation points.

    Args:
        xy_points: (Q, 2) collocation points
        W: (M, 2) hidden layer weights
        b: (M,) biases
        x_lo, x_hi, y_lo, y_hi: subdomain bounds

    Returns:
        V, dV_dx, dV_dy, d2V_dx2, d2V_dy2: each (Q, M)
    """
    scale_x = 2.0 / (x_hi - x_lo)
    scale_y = 2.0 / (y_hi - y_lo)
    x_norm = 2.0 * (xy_points[:, 0] - x_lo) / (x_hi - x_lo) - 1.0
    y_norm = 2.0 * (xy_points[:, 1] - y_lo) / (y_hi - y_lo) - 1.0

    z = x_norm[:, None] * W[None, :, 0] + y_norm[:, None] * W[None, :, 1] + b[None, :]
    V = jnp.tanh(z)
    sech2 = 1.0 - V**2
    wx = W[None, :, 0] * scale_x
    wy = W[None, :, 1] * scale_y

    dV_dx = sech2 * wx
    dV_dy = sech2 * wy
    d2V_dx2 = -2.0 * V * sech2 * wx**2
    d2V_dy2 = -2.0 * V * sech2 * wy**2

    return V, dV_dx, dV_dy, d2V_dx2, d2V_dy2


def solve_locelm_2d(
    pde_coeffs, source_fn, bc_fn, domain, Nx, Ny, Qx, Qy, M, R_m, seed=0
):
    """Solve a 2D linear PDE with Dirichlet BCs using locELM.

    PDE: a_xx * d²u/dx² + a_yy * d²u/dy² + a_0 * u = f(x,y)
    on [a1, b1] x [a2, b2] with u = bc_fn on boundary.

    C^1 continuity is imposed on all internal subdomain boundaries.
    """
    a1, b1, a2, b2 = domain
    a_xx, a_yy, a_0 = pde_coeffs

    x_bounds = jnp.linspace(a1, b1, Nx + 1)
    y_bounds = jnp.linspace(a2, b2, Ny + 1)

    N_e = Nx * Ny
    Qxy = Qx * Qy
    total_unknowns = N_e * M

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, N_e)

    subdomains = {}
    for m in range(Nx):
        for n in range(Ny):
            e = m * Ny + n
            x_lo = float(x_bounds[m])
            x_hi = float(x_bounds[m + 1])
            y_lo = float(y_bounds[n])
            y_hi = float(y_bounds[n + 1])
            W, b_vec = init_params_2d(keys[e], M, R_m)

            xx, yy = jnp.meshgrid(
                jnp.linspace(x_lo, x_hi, Qx),
                jnp.linspace(y_lo, y_hi, Qy),
                indexing="ij",
            )
            xy_pts = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

            V, dV_dx, dV_dy, d2V_dx2, d2V_dy2 = compute_basis_2d(
                xy_pts, W, b_vec, x_lo, x_hi, y_lo, y_hi
            )
            LV = a_xx * d2V_dx2 + a_yy * d2V_dy2 + a_0 * V

            subdomains[(m, n)] = {
                "W": W,
                "b": b_vec,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "xy_points": xy_pts,
                "LV": LV,
            }

    total_rows = N_e * (Qxy + 2 * Qx + 2 * Qy)
    A = jnp.zeros((total_rows, total_unknowns))
    rhs = jnp.zeros(total_rows)
    row = 0

    def eidx(m, n):
        return m * Ny + n

    # --- PDE residual: N_e * Qxy rows ---
    for m in range(Nx):
        for n in range(Ny):
            e = eidx(m, n)
            sd = subdomains[(m, n)]
            A = A.at[row : row + Qxy, e * M : (e + 1) * M].set(sd["LV"])
            rhs = rhs.at[row : row + Qxy].set(jax.vmap(source_fn)(sd["xy_points"]))
            row += Qxy

    # --- Boundary conditions ---
    def _add_bc(A, rhs, row, sd, e, xy_bc):
        n_pts = xy_bc.shape[0]
        V_bc = compute_basis_2d(
            xy_bc, sd["W"], sd["b"], sd["x_lo"], sd["x_hi"], sd["y_lo"], sd["y_hi"]
        )[0]
        A = A.at[row : row + n_pts, e * M : (e + 1) * M].set(V_bc)
        rhs = rhs.at[row : row + n_pts].set(jax.vmap(bc_fn)(xy_bc))
        return A, rhs, row + n_pts

    # Left x=a1
    for n in range(Ny):
        sd = subdomains[(0, n)]
        y_pts = jnp.linspace(sd["y_lo"], sd["y_hi"], Qy)
        xy_bc = jnp.stack([jnp.full(Qy, a1), y_pts], axis=1)
        A, rhs, row = _add_bc(A, rhs, row, sd, eidx(0, n), xy_bc)

    # Right x=b1
    for n in range(Ny):
        sd = subdomains[(Nx - 1, n)]
        y_pts = jnp.linspace(sd["y_lo"], sd["y_hi"], Qy)
        xy_bc = jnp.stack([jnp.full(Qy, b1), y_pts], axis=1)
        A, rhs, row = _add_bc(A, rhs, row, sd, eidx(Nx - 1, n), xy_bc)

    # Bottom y=a2
    for m in range(Nx):
        sd = subdomains[(m, 0)]
        x_pts = jnp.linspace(sd["x_lo"], sd["x_hi"], Qx)
        xy_bc = jnp.stack([x_pts, jnp.full(Qx, a2)], axis=1)
        A, rhs, row = _add_bc(A, rhs, row, sd, eidx(m, 0), xy_bc)

    # Top y=b2
    for m in range(Nx):
        sd = subdomains[(m, Ny - 1)]
        x_pts = jnp.linspace(sd["x_lo"], sd["x_hi"], Qx)
        xy_bc = jnp.stack([x_pts, jnp.full(Qx, b2)], axis=1)
        A, rhs, row = _add_bc(A, rhs, row, sd, eidx(m, Ny - 1), xy_bc)

    # --- C^1 continuity ---
    # Vertical internal boundaries x = X_{m+1}
    for m in range(Nx - 1):
        xb = float(x_bounds[m + 1])
        for n in range(Ny):
            sd_l = subdomains[(m, n)]
            sd_r = subdomains[(m + 1, n)]
            e_l, e_r = eidx(m, n), eidx(m + 1, n)

            y_pts = jnp.linspace(sd_l["y_lo"], sd_l["y_hi"], Qy)
            xy_bnd = jnp.stack([jnp.full(Qy, xb), y_pts], axis=1)

            V_l, dVx_l, *_ = compute_basis_2d(
                xy_bnd,
                sd_l["W"],
                sd_l["b"],
                sd_l["x_lo"],
                sd_l["x_hi"],
                sd_l["y_lo"],
                sd_l["y_hi"],
            )
            V_r, dVx_r, *_ = compute_basis_2d(
                xy_bnd,
                sd_r["W"],
                sd_r["b"],
                sd_r["x_lo"],
                sd_r["x_hi"],
                sd_r["y_lo"],
                sd_r["y_hi"],
            )

            # C^0
            A = A.at[row : row + Qy, e_l * M : (e_l + 1) * M].set(V_l)
            A = A.at[row : row + Qy, e_r * M : (e_r + 1) * M].set(-V_r)
            row += Qy

            # C^1 (du/dx)
            A = A.at[row : row + Qy, e_l * M : (e_l + 1) * M].set(dVx_l)
            A = A.at[row : row + Qy, e_r * M : (e_r + 1) * M].set(-dVx_r)
            row += Qy

    # Horizontal internal boundaries y = Y_{n+1}
    for n in range(Ny - 1):
        yb = float(y_bounds[n + 1])
        for m in range(Nx):
            sd_b = subdomains[(m, n)]
            sd_t = subdomains[(m, n + 1)]
            e_b, e_t = eidx(m, n), eidx(m, n + 1)

            x_pts = jnp.linspace(sd_b["x_lo"], sd_b["x_hi"], Qx)
            xy_bnd = jnp.stack([x_pts, jnp.full(Qx, yb)], axis=1)

            V_b, _, dVy_b, *_ = compute_basis_2d(
                xy_bnd,
                sd_b["W"],
                sd_b["b"],
                sd_b["x_lo"],
                sd_b["x_hi"],
                sd_b["y_lo"],
                sd_b["y_hi"],
            )
            V_t, _, dVy_t, *_ = compute_basis_2d(
                xy_bnd,
                sd_t["W"],
                sd_t["b"],
                sd_t["x_lo"],
                sd_t["x_hi"],
                sd_t["y_lo"],
                sd_t["y_hi"],
            )

            # C^0
            A = A.at[row : row + Qx, e_b * M : (e_b + 1) * M].set(V_b)
            A = A.at[row : row + Qx, e_t * M : (e_t + 1) * M].set(-V_t)
            row += Qx

            # C^1 (du/dy)
            A = A.at[row : row + Qx, e_b * M : (e_b + 1) * M].set(dVy_b)
            A = A.at[row : row + Qx, e_t * M : (e_t + 1) * M].set(-dVy_t)
            row += Qx

    beta, _, _, _ = scipy.linalg.lstsq(
        np.array(A), np.array(rhs), lapack_driver="gelsd"
    )
    return jnp.array(beta), subdomains


def evaluate_solution_2d(xy_eval, beta, subdomains, Nx, Ny, M):
    """Evaluate the 2D locELM solution at arbitrary points."""
    u_eval = jnp.zeros(xy_eval.shape[0])

    for m in range(Nx):
        for n in range(Ny):
            sd = subdomains[(m, n)]
            in_x = (xy_eval[:, 0] >= sd["x_lo"]) & (
                xy_eval[:, 0] < sd["x_hi"]
                if m < Nx - 1
                else xy_eval[:, 0] <= sd["x_hi"]
            )
            in_y = (xy_eval[:, 1] >= sd["y_lo"]) & (
                xy_eval[:, 1] < sd["y_hi"]
                if n < Ny - 1
                else xy_eval[:, 1] <= sd["y_hi"]
            )
            mask = in_x & in_y

            x_norm = (
                2.0 * (xy_eval[:, 0] - sd["x_lo"]) / (sd["x_hi"] - sd["x_lo"]) - 1.0
            )
            y_norm = (
                2.0 * (xy_eval[:, 1] - sd["y_lo"]) / (sd["y_hi"] - sd["y_lo"]) - 1.0
            )
            z = (
                x_norm[:, None] * sd["W"][None, :, 0]
                + y_norm[:, None] * sd["W"][None, :, 1]
                + sd["b"][None, :]
            )
            V = jnp.tanh(z)

            e = m * Ny + n
            u_s = V @ beta[e * M : (e + 1) * M]
            u_eval = u_eval + u_s * mask

    return u_eval
