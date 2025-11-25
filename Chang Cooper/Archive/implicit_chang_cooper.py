import numpy as np

# Constants
sigma = 0.001
beta = 100
lamda = 0.6
alpha = 0.1
v0 = 0.01
vg = 0
psi = np.inf
D = 0.00001

t0 = 0.0
tf = 5000
steps = 100000

# --------- equations------------



# ---------- helpers ----------

def thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system:
      a[j]*x[j-1] + b[j]*x[j] + c[j]*x[j+1] = d[j]
    with a[0]=0, c[-1]=0.
    """
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x  = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for j in range(1, n):
        denom = b[j] - a[j] * cp[j-1]
        cp[j] = c[j] / denom if j < n-1 else 0.0
        dp[j] = (d[j] - a[j] * dp[j-1]) / denom

    x[-1] = dp[-1]
    for j in range(n-2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j+1]
    return x


def delta_cc(Pe):
    if abs(Pe) < 1e-12:
        return 0.5
    return 1.0/Pe - 1.0/(np.exp(Pe) - 1.0)


# ---------- Chang–Cooper implicit phi sweep (1D line) ----------

def cc_phi_implicit_line(f, A, D, dphi, dt, bc="periodic"):
    """
    One implicit Chang–Cooper step in phi for a single (x,z) line.

    f  : (Nphi,)
    A  : (Nphi,) drift mu_phi at this (x,z)
    D  : (Nphi,) diffusion coefficient = 0.5*var^2
    bc : "periodic" or "noflux"
    """
    N = len(f)

    # faces j+1/2 built from neighbor pairs
    A_half = np.zeros(N)   # face between j and j+1, for j=0..N-1 (periodic wrap)
    D_half = np.zeros(N)
    delta  = np.zeros(N)

    for j in range(N):
        jp = (j + 1) % N
        A_half[j] = 0.5 * (A[j] + A[jp])
        D_half[j] = 0.5 * (D[j] + D[jp])
        Pe = A_half[j] * dphi / D_half[j]
        delta[j] = delta_cc(Pe)

    # express face flux as J_{j+1/2} = p_{j+1/2} f_{j+1} + q_{j+1/2} f_j
    p = np.zeros(N)
    q = np.zeros(N)
    for j in range(N):
        p[j] = A_half[j] * (1.0 - delta[j]) - D_half[j] / dphi
        q[j] = A_half[j] * delta[j]        + D_half[j] / dphi

    # build tridiagonal system
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    rhs = f.copy()

    for j in range(N):
        jm = (j - 1) % N
        # coefficients from divergence -(J_{j+1/2}-J_{j-1/2})/dphi
        a[j] = -dt * q[jm] / dphi          # multiplies f_{j-1}
        b[j] = 1.0 - dt * (p[jm] - q[j]) / dphi
        c[j] =  dt * p[j] / dphi          # multiplies f_{j+1}

    if bc == "noflux":
        # enforce J_{1/2}=J_{N-1/2}=0 by one-sided closure
        # simplest: pin boundary faces to zero -> modify first/last rows
        a[0] = 0.0
        c[-1] = 0.0

    if bc == "periodic":
        # periodic system is cyclic; solve by converting to full matrix and using np.linalg.solve
        # (still fine for moderate Nphi; replace by cyclic tridiagonal solver if needed)
        M = np.zeros((N, N))
        for j in range(N):
            M[j, j] = b[j]
            if j > 0:     M[j, j-1] = a[j]
            if j < N-1:   M[j, j+1] = c[j]
        # wrap couplings
        M[0, -1]  = a[0]
        M[-1, 0]  = c[-1]
        return np.linalg.solve(M, rhs)

    return thomas_solve(a, b, c, rhs)


# ---------- Implicit upwind advection sweep (1D line) ----------

def upwind_advection_implicit_line(f, u, dx, dt, bc="periodic"):
    """
    Implicit FV upwind advection in one coordinate:
      f_t = -∂x(u f)
    f : (N,)
    u : (N,) cell-centered velocity along this line
    """
    N = len(f)

    # face velocities
    u_half = np.zeros(N)
    for i in range(N):
        ip = (i + 1) % N
        u_half[i] = 0.5 * (u[i] + u[ip])

    # Upwind flux F_{i+1/2} = u^+ f_i + u^- f_{i+1}
    p = np.minimum(u_half, 0.0)  # multiplies f_{i+1}
    q = np.maximum(u_half, 0.0)  # multiplies f_i

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    rhs = f.copy()

    for i in range(N):
        im = (i - 1) % N
        a[i] = -dt * q[im] / dx
        b[i] = 1.0 - dt * (p[im] - q[i]) / dx
        c[i] =  dt * p[i] / dx

    if bc == "periodic":
        M = np.zeros((N, N))
        for i in range(N):
            M[i, i] = b[i]
            if i > 0:   M[i, i-1] = a[i]
            if i < N-1: M[i, i+1] = c[i]
        M[0, -1] = a[0]
        M[-1, 0] = c[-1]
        return np.linalg.solve(M, rhs)

    return thomas_solve(a, b, c, rhs)


# ---------- full 3D step: Strang-split implicit sweeps ----------

def fokker_planck_step_implicit(
    f, t, dt,
    x_grid, z_grid, phi_grid,
    mu_x, mu_z, mu_phi, var_fun,
    bc_x="periodic", bc_z="periodic", bc_phi="periodic"
):
    """
    One Strang-splitting implicit step for the 3D FP:
      f_t = -∂x(mu_x f) -∂z(mu_z f)
            -∂phi(mu_phi f) + ∂phi(D ∂phi f),
      D = 0.5*var^2

    f shape: (Nx, Nz, Nphi)
    """
    Nx, Nz, Np = f.shape
    dx   = x_grid[1] - x_grid[0]
    dz   = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]

    # precompute drifts/diffusion at time t (explicit-in-coefficients)
    MUx   = np.zeros_like(f)
    MUz   = np.zeros_like(f)
    MUphi = np.zeros_like(f)
    Dphi  = np.zeros_like(f)

    for i, x in enumerate(x_grid):
        for k, z in enumerate(z_grid):
            for j, ph in enumerate(phi_grid):
                vec = np.array([x, z, ph])
                MUx[i,k,j]   = mu_x(vec, t)
                MUz[i,k,j]   = mu_z(vec, t)
                MUphi[i,k,j] = mu_phi(vec, t)
                Dphi[i,k,j]  = 0.5 * var_fun(vec, t)**2

    # ----- z half-step -----
    f1 = f.copy()
    for i in range(Nx):
        for j in range(Np):
            line = f1[i, :, j]
            uline = MUz[i, :, j]
            f1[i, :, j] = upwind_advection_implicit_line(line, uline, dz, 0.5*dt, bc=bc_z)

    # ----- x half-step -----
    f2 = f1.copy()
    for k in range(Nz):
        for j in range(Np):
            line = f2[:, k, j]
            uline = MUx[:, k, j]
            f2[:, k, j] = upwind_advection_implicit_line(line, uline, dx, 0.5*dt, bc=bc_x)

    # ----- phi full-step (Chang–Cooper implicit) -----
    f3 = f2.copy()
    for i in range(Nx):
        for k in range(Nz):
            line_f = f3[i, k, :]
            line_A = MUphi[i, k, :]
            line_D = Dphi[i, k, :]
            f3[i, k, :] = cc_phi_implicit_line(line_f, line_A, line_D, dphi, dt, bc=bc_phi)

    # ----- x half-step -----
    f4 = f3.copy()
    for k in range(Nz):
        for j in range(Np):
            line = f4[:, k, j]
            uline = MUx[:, k, j]
            f4[:, k, j] = upwind_advection_implicit_line(line, uline, dx, 0.5*dt, bc=bc_x)

    # ----- z half-step -----
    f5 = f4.copy()
    for i in range(Nx):
        for j in range(Np):
            line = f5[i, :, j]
            uline = MUz[i, :, j]
            f5[i, :, j] = upwind_advection_implicit_line(line, uline, dz, 0.5*dt, bc=bc_z)

    return f5


# ---------- time integrator ----------

def solve_fp(
    f0, t0, tf, Nt,
    x_grid, z_grid, phi_grid,
    mu_x, mu_z, mu_phi, var_fun,
    bc_x="periodic", bc_z="periodic", bc_phi="periodic"
):
    dt = (tf - t0) / Nt
    f = f0.copy()
    t = t0
    out = np.zeros((Nt+1,) + f.shape)
    out[0] = f

    for n in range(Nt):
        f = fokker_planck_step_implicit(
            f, t, dt,
            x_grid, z_grid, phi_grid,
            mu_x, mu_z, mu_phi, var_fun,
            bc_x=bc_x, bc_z=bc_z, bc_phi=bc_phi
        )
        t += dt
        out[n+1] = f

    return out
