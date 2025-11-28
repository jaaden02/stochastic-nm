import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ========== Model Parameters ==========
sigma = 0.001
beta = 100
lamda = 0.6
alpha = 0.1
v0 = 0.01
vg = 0
psi = np.inf
D0 = 0.0001

# ========== Vectorized Drift and Diffusion Functions ==========

def mu_x_vec(X, Z, PHI, t):
    """Vectorized x-direction drift"""
    return alpha*np.exp(Z)*np.cos(X-t) + v0*np.sin(PHI) + sigma*(beta+Z)

def mu_z_vec(X, Z, PHI, t):
    """Vectorized z-direction drift"""
    return alpha*np.exp(Z)*np.sin(X-t) + v0*np.cos(PHI) - vg

def mu_phi_vec(X, Z, PHI, t):
    """Vectorized phi-direction drift"""
    return (lamda*alpha*np.exp(Z)*np.cos(X-t+2*PHI) - 
            1/(2*psi)*np.sin(PHI) + 
            sigma/2*(1+lamda*np.cos(2*PHI)))

def var_vec(X, Z, PHI, t):
    """Vectorized noise intensity"""
    # TODO: Is it not like PHI
    return D0 * np.ones_like(X)

# ========== Helper Functions ==========

def delta_cc_vec(Pe):
    """Vectorized Chang-Cooper delta function"""
    delta = np.zeros_like(Pe)
    
    # Small Pe: use limiting value
    small = np.abs(Pe) < 1e-10
    delta[small] = 0.5
    
    # Normal Pe
    large = ~small
    delta[large] = 1.0/Pe[large] - 1.0/(np.exp(Pe[large]) - 1.0)
    
    return delta

def apply_periodic_bc(arr, axis):
    """Apply periodic boundary conditions along an axis"""
    return np.roll(arr, 0, axis=axis)  # Identity, already periodic in indexing

def thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system:
      a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    with a[0]=0, c[-1]=0.
    """
    n = len(b)
    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)
    x  = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom

    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

# ========== Vectorized Advection Line ==========

def implicit_upwind_advection_line(f_line, u_line, dt, dx, bc):
    """
    Implicit finite-volume upwind step for:
        f_t = -∂x( u f )
    on a 1D line.

    f_line : (N,)
    u_line : (N,)  cell-centered velocity
    bc     : "open" (x) or "noflux" (z)
    """
    N = f_line.size
    f_line = f_line.astype(float, copy=False)
    u_line = u_line.astype(float, copy=False)

    # interface velocities u_{i+1/2}
    u_half = 0.5 * (u_line[:-1] + u_line[1:])      # size N-1
    u_half_p = np.maximum(u_half, 0.0)
    u_half_m = np.minimum(u_half, 0.0)

    a = np.zeros(N, dtype=float)
    b = np.zeros(N, dtype=float)
    c = np.zeros(N, dtype=float)
    rhs = f_line.copy()
    lam = dt / dx

    if bc == "open":
        # --- left boundary i=0: ghost = 0 outside, use u_L = u[0]
        uL = u_line[0]
        uL_m = min(uL, 0.0)
        # Face at 1/2 uses u_half[0]
        uhp0_p = u_half_p[0]
        uhp0_m = u_half_m[0]

        a[0] = 0.0
        b[0] = 1.0 + lam * (uhp0_p - uL_m)
        c[0] = lam * uhp0_m

        # --- interior 1..N-2
        for i in range(1, N-1):
            um_p = u_half_p[i-1]   # u_{i-1/2}^+
            um_m = u_half_m[i-1]   # u_{i-1/2}^-
            up_p = u_half_p[i]     # u_{i+1/2}^+
            up_m = u_half_m[i]     # u_{i+1/2}^-

            a[i] = -lam * um_p
            b[i] = 1.0 + lam * (up_p - um_m)
            c[i] = lam * up_m

        # --- right boundary i=N-1: ghost = 0 outside, use uR = u[-1]
        uR = u_line[-1]
        uR_p = max(uR, 0.0)
        # upwind face at N-3/2 uses u_half[N-2]
        um_p = u_half_p[N-2]
        um_m = u_half_m[N-2]

        a[N-1] = -lam * um_p
        b[N-1] = 1.0 + lam * (uR_p - um_m)
        c[N-1] = 0.0

    elif bc == "noflux":
        # --- left boundary: F_{-1/2}=0
        # f0^{n+1} + lam * F_{1/2} = f0^n
        uhp0_p = u_half_p[0]
        uhp0_m = u_half_m[0]
        a[0] = 0.0
        b[0] = 1.0 + lam * uhp0_p
        c[0] = lam * uhp0_m

        # --- interior 1..N-2
        for i in range(1, N-1):
            um_p = u_half_p[i-1]
            um_m = u_half_m[i-1]
            up_p = u_half_p[i]
            up_m = u_half_m[i]

            a[i] = -lam * um_p
            b[i] = 1.0 + lam * (up_p - um_m)
            c[i] = lam * up_m

        # --- right boundary: F_{N-1/2}=0
        um_p = u_half_p[N-2]
        um_m = u_half_m[N-2]
        a[N-1] = -lam * um_p
        b[N-1] = 1.0 + lam * um_m
        c[N-1] = 0.0

    else:
        raise ValueError("bc must be 'open' or 'noflux'")

    return thomas_solve(a, b, c, rhs)


# ========== Vectorized Fokker-Planck Step ==========

def cc_phi_implicit_line(f_line, A_line, D_line, dt, dphi):
    """
    Implicit Chang–Cooper step in phi for a single (x,z) line, periodic in phi.

    f_line : (Nphi,)
    A_line : (Nphi,)  drift mu_phi
    D_line : (Nphi,)  diffusion coef = 0.5*var^2
    """
    N = f_line.size
    eps = 1e-16

    f_line = f_line.astype(float, copy=False)
    A_line = A_line.astype(float, copy=False)
    D_line = D_line.astype(float, copy=False)

    # interfaces j+1/2 (periodic)
    A_shift = np.roll(A_line, -1)
    D_shift = np.roll(D_line, -1)
    A_half = 0.5 * (A_line + A_shift)
    D_half = 0.5 * (D_line + D_shift)

    Pe = A_half * dphi / (D_half + eps)
    delta = delta_cc_vec(Pe)  # your vectorized delta

    # J_{j+1/2} = p_j f_{j+1} + q_j f_j
    p = A_half * (1.0 - delta) - D_half / dphi
    q = A_half * delta          + D_half / dphi

    # Build dense cyclic matrix M x = rhs
    M = np.zeros((N, N), dtype=float)
    rhs = f_line.copy()
    lam = dt / dphi

    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N

        a_j = -lam * q[jm]
        b_j =  1.0 + lam * (q[j] - p[jm])
        c_j =  lam * p[j]

        M[j, jm] += a_j
        M[j, j]  += b_j
        M[j, jp] += c_j

    return np.linalg.solve(M, rhs)

def fokker_planck_step_implicit_full(f, X, Z, PHI, t, dt, dx, dz, dphi):
    """
    Fully implicit ADI step:
      - implicit in x (open BC)
      - implicit in z (noflux BC)
      - implicit Chang–Cooper in phi (periodic)
    """
    Nx, Nz, Nphi = f.shape

    # coefficients at time t (frozen)
    MU_X   = mu_x_vec(X, Z, PHI, t)
    MU_Z   = mu_z_vec(X, Z, PHI, t)
    MU_PHI = mu_phi_vec(X, Z, PHI, t)
    VAR    = var_vec(X, Z, PHI, t)
    D      = 0.5 * VAR**2

    dt_half = 0.5 * dt

    # Strang: x(Δt/2) -> z(Δt/2) -> φ(Δt) -> z(Δt/2) -> x(Δt/2)
    f1 = sweep_x_implicit(f, MU_X, dt_half, dx)
    f2 = sweep_z_implicit(f1, MU_Z, dt_half, dz)
    f3 = sweep_phi_implicit(f2, MU_PHI, D, dt, dphi)
    f4 = sweep_z_implicit(f3, MU_Z, dt_half, dz)
    f5 = sweep_x_implicit(f4, MU_X, dt_half, dx)

    # clip negative due to roundoff
    f5 = np.maximum(f5, 0.0)
    return f5

# ========== Strang Splitting ==========

def sweep_x_implicit(f, MU_X, dt_half, dx):
    """
    Implicit x-sweep (open BC) on entire 3D array.
    """
    Nx, Nz, Nphi = f.shape
    out = np.empty_like(f)

    for k in range(Nz):
        for j in range(Nphi):
            f_line = f[:, k, j]
            u_line = MU_X[:, k, j]
            out[:, k, j] = implicit_upwind_advection_line(
                f_line, u_line, dt_half, dx, bc="open"
            )
    return out

def sweep_z_implicit(f, MU_Z, dt_half, dz):
    """
    Implicit z-sweep (noflux BC) on entire 3D array.
    """
    Nx, Nz, Nphi = f.shape
    out = np.empty_like(f)

    for i in range(Nx):
        for j in range(Nphi):
            f_line = f[i, :, j]
            u_line = MU_Z[i, :, j]
            out[i, :, j] = implicit_upwind_advection_line(
                f_line, u_line, dt_half, dz, bc="noflux"
            )
    return out

def sweep_phi_implicit(f, MU_PHI, D, dt, dphi):
    """
    Implicit phi-sweep (Chang–Cooper, periodic) on entire 3D array.
    """
    Nx, Nz, Nphi = f.shape
    out = np.empty_like(f)

    for i in range(Nx):
        for k in range(Nz):
            f_line = f[i, k, :]
            A_line = MU_PHI[i, k, :]
            D_line = D[i, k, :]
            out[i, k, :] = cc_phi_implicit_line(f_line, A_line, D_line, dt, dphi)
    return out

# ========== Time Integration ==========

def solve_fokker_planck_implicit_full(f0, t_array, x_grid, z_grid, phi_grid,
                                      verbose=True, live_plot=False, plot_interval=50):
    """
    Fully implicit ADI Fokker–Planck solver:
      x,z implicit upwind, phi implicit Chang–Cooper.
    """
    Nt = len(t_array)
    Nx, Nz, Nphi = f0.shape

    dx   = x_grid[1]  - x_grid[0]
    dz   = z_grid[1]  - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]

    X, Z, PHI = np.meshgrid(x_grid, z_grid, phi_grid, indexing='ij')

    solution = np.zeros((Nt, Nx, Nz, Nphi))
    solution[0] = f0.copy()
    f = f0.copy()

    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Live Fokker-Planck Evolution (full implicit ADI)', fontsize=14)

    if verbose:
        print("Starting fully implicit ADI time integration...")

    for n in range(Nt - 1):
        dt = t_array[n+1] - t_array[n]
        t  = t_array[n]

        f = fokker_planck_step_implicit_full(f, X, Z, PHI, t, dt, dx, dz, dphi)
        solution[n+1] = f

        if np.any(np.isnan(f)):
            print(f"NaN detected at step {n+1}!")
            break

        if verbose and ((n+1) % 100 == 0 or n == 0):
            total_prob = np.sum(f) * dx * dz * dphi
            print(f"Step {n+1}/{Nt-1}, t={t_array[n+1]:.4f}, "
                  f"∫f={total_prob:.6f}, min={f.min():.3e}, max={f.max():.3e}")

        if live_plot and (n+1) % plot_interval == 0:
            plot_state(f, x_grid, z_grid, phi_grid, t_array[n+1], axes, fig)

    if live_plot:
        plt.ioff()
        plt.show()

    return solution

def check_cfl_condition(x_grid, z_grid, phi_grid, dt, t=0):
    """Check CFL stability condition"""
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    D_max = 0.5 * 0.3**2  # var_fun returns 0.3
    
    if D_max > 0:
        dt_max_x = dx**2 / (2 * D_max)
        dt_max_z = dz**2 / (2 * D_max)
        dt_max_phi = dphi**2 / (2 * D_max)
        dt_max = min(dt_max_x, dt_max_z, dt_max_phi)
        
        print(f"CFL check: dt={dt:.4e}, dt_max={dt_max:.4e}")
        if dt > 0.5 * dt_max:
            print(f"Warning: dt may be too large for stability!")
        else:
            print(f"✓ CFL condition satisfied")
    
    return dt

# MARK:  Plotting Functions
# ========== Plotting Functions ==========

def plot_state(f, x_grid, z_grid, phi_grid, t, axes, fig):
    """Update live plot with current state"""
    for ax in axes.flat:
        ax.clear()
    
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    # 1. Marginal in (x, z) - integrate over phi
    f_xz = np.sum(f, axis=2) * dphi
    im1 = axes[0, 0].contourf(x_grid, z_grid, f_xz.T, levels=20, cmap='viridis')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('z')
    axes[0, 0].set_title(f'Marginal f(x,z) at t={t:.3f}')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Marginal in (x, phi) - integrate over z
    f_xphi = np.sum(f, axis=1) * dz
    im2 = axes[0, 1].contourf(x_grid, phi_grid, f_xphi.T, levels=20, cmap='plasma')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('φ')
    axes[0, 1].set_title(f'Marginal f(x,φ)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Marginal in (z, phi) - integrate over x
    f_zphi = np.sum(f, axis=0) * dx
    im3 = axes[1, 0].contourf(z_grid, phi_grid, f_zphi.T, levels=20, cmap='coolwarm')
    axes[1, 0].set_xlabel('z')
    axes[1, 0].set_ylabel('φ')
    axes[1, 0].set_title(f'Marginal f(z,φ)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. 1D marginals
    f_x = np.sum(f, axis=(1, 2)) * dz * dphi
    f_z = np.sum(f, axis=(0, 2)) * dx * dphi
    f_phi = np.sum(f, axis=(0, 1)) * dx * dz
    
    axes[1, 1].plot(x_grid, f_x, 'b-', label='f(x)', linewidth=2)
    axes[1, 1].plot(z_grid, f_z, 'r-', label='f(z)', linewidth=2)
    axes[1, 1].plot(phi_grid, f_phi, 'g-', label='f(φ)', linewidth=2)
    axes[1, 1].set_xlabel('Coordinate')
    axes[1, 1].set_ylabel('Probability density')
    axes[1, 1].set_title('1D Marginals')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.pause(0.01)

def plot_summary(solution, t_array, x_grid, z_grid, phi_grid, save_path=None):
    """Create comprehensive summary plots after simulation"""
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    fig = plt.figure(figsize=(16, 10))
    
    # Select key time points
    time_indices = [0, len(t_array)//4, len(t_array)//2, 3*len(t_array)//4, -1]
    
    for idx, ti in enumerate(time_indices):
        f = solution[ti]
        t = t_array[ti]
        
        # Plot marginal in (x, z)
        ax = plt.subplot(3, 5, idx+1)
        f_xz = np.sum(f, axis=2) * dphi
        im = ax.contourf(x_grid, z_grid, f_xz.T, levels=15, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f't={t:.3f}')
        plt.colorbar(im, ax=ax)
        
        # Plot marginal in (x, phi)
        ax = plt.subplot(3, 5, idx+6)
        f_xphi = np.sum(f, axis=1) * dz
        im = ax.contourf(x_grid, phi_grid, f_xphi.T, levels=15, cmap='plasma')
        ax.set_xlabel('x')
        ax.set_ylabel('φ')
        plt.colorbar(im, ax=ax)
        
        # Plot 1D marginals
        ax = plt.subplot(3, 5, idx+11)
        f_x = np.sum(f, axis=(1, 2)) * dz * dphi
        f_z = np.sum(f, axis=(0, 2)) * dx * dphi
        f_phi = np.sum(f, axis=(0, 1)) * dx * dz
        ax.plot(x_grid, f_x, 'b-', label='f(x)', alpha=0.7)
        ax.plot(z_grid, f_z, 'r-', label='f(z)', alpha=0.7)
        ax.plot(phi_grid, f_phi, 'g-', label='f(φ)', alpha=0.7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Coordinate')
    
    plt.suptitle('Fokker-Planck Evolution Summary', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to: {save_path}")
    
    plt.show()
    
    return fig

# MARK: Example Usage
# ========== Example Usage ==========

if __name__ == "__main__":
    import time
    
    # Grid setup
    Nx, Nz, Nphi = 30, 30, 50
    x_grid = np.linspace(-20, 20, Nx)
    z_grid = np.linspace(-2, 2, Nz)
    phi_grid = np.linspace(0, 2*np.pi, Nphi)
    
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    print("="*60)
    print("VECTORIZED 3D FOKKER-PLANCK SOLVER")
    print("="*60)
    print(f"Grid: {Nx}×{Nz}×{Nphi} = {Nx*Nz*Nphi:,} cells")
    print(f"Resolution: dx={dx:.3f}, dz={dz:.3f}, dφ={dphi:.3f}")
    
    # Initial condition: localized Gaussian
    X, Z, PHI = np.meshgrid(x_grid, z_grid, phi_grid, indexing='ij')
    x0, z0, phi0 = 0.0, -1.0, np.pi
    sigma_init = 0.5
    
    f0 = np.exp(-((X-x0)**2 + (Z-z0)**2 + (PHI-phi0)**2)/(2*sigma_init**2))
    f0 = f0 / (np.sum(f0) * dx * dz * dphi)
    
    print(f"Initial ∫f = {np.sum(f0)*dx*dz*dphi:.6f}")
    print()
    
    # Time setup - use smaller dt for stability
    t_final = 2.0
    dt = 0.0005  # Reduced from 0.001 for better stability
    Nt = int(t_final / dt) + 1
    t_array = np.linspace(0, t_final, Nt)
    
    print(f"Time: t ∈ [0, {t_final}], dt={dt:.4e}, Nt={Nt}")
    check_cfl_condition(x_grid, z_grid, phi_grid, dt)
    print()
    
    # Solve with live plotting
    start_time = time.time()
    
    # Set live_plot=True to see real-time updates (slower)
    # Set live_plot=False for faster computation, then plot after
    use_live_plot = False  # Change to True for live visualization
    
    solution = solve_fokker_planck_implicit_full(
        f0, t_array, x_grid, z_grid, phi_grid,
        verbose=True, live_plot=False
    )


    elapsed = time.time() - start_time
    
    print()
    print("="*60)
    print(f"✓ Done! Solution shape: {solution.shape}")
    print(f"Final ∫f = {np.sum(solution[-1])*dx*dz*dphi:.6f}")
    print(f"Computation time: {elapsed:.2f} seconds")
    print(f"   ({elapsed/Nt:.4f} sec/step, {Nx*Nz*Nphi*Nt/elapsed/1e6:.2f} Mcells/sec)")
    print("="*60)
    
    # Create summary plots
    print("\nGenerating summary plots...")
    plot_summary(solution, t_array, x_grid, z_grid, phi_grid, 
                 save_path='fokker_planck_summary.png')