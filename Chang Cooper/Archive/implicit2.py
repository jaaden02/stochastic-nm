# ========== Vectorized Drift and Diffusion Functions ==========
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

# ========== Vectorized Fokker-Planck Step ==========

def cc_phi_implicit_line(f_line, A_line, D_line, dt, dphi):
    """
    Implicit Chang–Cooper step in phi for a single (x,z) line.

    f_line : (Nphi,)   current density at fixed (x,z)
    A_line : (Nphi,)   drift mu_phi at this (x,z)
    D_line : (Nphi,)   diffusion coef = 0.5 * var^2 at this (x,z)
    dt, dphi : floats
    Periodic in phi.
    """
    N = f_line.size
    eps = 1e-16

    # interfaces j+1/2 (periodic)
    A_shift = np.roll(A_line, -1)
    D_shift = np.roll(D_line, -1)
    A_half = 0.5 * (A_line + A_shift)
    D_half = 0.5 * (D_line + D_shift)

    Pe = A_half * dphi / (D_half + eps)
    delta = delta_cc_vec(Pe)

    # J_{j+1/2} = p_j f_{j+1} + q_j f_j
    p = A_half * (1.0 - delta) - D_half / dphi
    q = A_half * delta          + D_half / dphi

    # Build dense cyclic matrix M x = rhs
    M = np.zeros((N, N), dtype=float)
    rhs = f_line.copy()

    factor = dt / dphi

    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N

        # coefficients from: f^{n+1}_j + dt/dphi (J_{j+1/2} - J_{j-1/2}) = f_tilde_j
        # => a_j f_{j-1} + b_j f_j + c_j f_{j+1} = rhs_j
        a_j = -factor * q[jm]                       # multiplies f_{j-1}
        b_j = 1.0 + factor * (q[j] - p[jm])         # multiplies f_j
        c_j =  factor * p[j]                        # multiplies f_{j+1}

        M[j, jm] += a_j
        M[j, j]  += b_j
        M[j, jp] += c_j

    # Solve
    return np.linalg.solve(M, rhs)

def fokker_planck_step_implicit_phi(f, X, Z, PHI, t, dt, dx, dz, dphi,
                                     bc_x='open', bc_z='noflux', bc_phi='periodic'):
    """
    Semi-implicit FP step:
      - x, z directions: explicit upwind (vectorized) as in your original code
      - phi direction: implicit Chang–Cooper with periodic BC
    """
    Nx, Nz, Nphi = f.shape

    # ==== coefficients at time t (frozen during step) ====
    MU_X   = mu_x_vec(X, Z, PHI, t)
    MU_Z   = mu_z_vec(X, Z, PHI, t)
    MU_PHI = mu_phi_vec(X, Z, PHI, t)
    VAR    = var_vec(X, Z, PHI, t)
    D      = 0.5 * VAR**2

    # ----------------------------------------------------
    # 1) Explicit x-direction fluxes (unchanged)
    # ----------------------------------------------------
    MU_X_face_p = 0.5 * (MU_X + np.roll(MU_X, -1, axis=0))
    F_x_p = np.where(MU_X_face_p >= 0,
                     MU_X_face_p * f,
                     MU_X_face_p * np.roll(f, -1, axis=0))
    MU_X_face_m = 0.5 * (np.roll(MU_X, 1, axis=0) + MU_X)
    F_x_m = np.where(MU_X_face_m >= 0,
                     MU_X_face_m * np.roll(f, 1, axis=0),
                     MU_X_face_m * f)

    if bc_x == 'open':
        F_x_m[0, :, :]  = np.minimum(0, MU_X[0, :, :])  * f[0, :, :]
        F_x_p[-1, :, :] = np.maximum(0, MU_X[-1, :, :]) * f[-1, :, :]
    elif bc_x == 'noflux':
        F_x_m[0, :, :]  = 0.0
        F_x_p[-1, :, :] = 0.0

    div_x = -(F_x_p - F_x_m) / dx

    # ----------------------------------------------------
    # 2) Explicit z-direction fluxes (unchanged, "one-sided / noflux")
    # ----------------------------------------------------
    MU_Z_face_p = 0.5 * (MU_Z + np.roll(MU_Z, -1, axis=1))
    F_z_p = np.where(MU_Z_face_p >= 0,
                     MU_Z_face_p * f,
                     MU_Z_face_p * np.roll(f, -1, axis=1))
    MU_Z_face_m = 0.5 * (np.roll(MU_Z, 1, axis=1) + MU_Z)
    F_z_m = np.where(MU_Z_face_m >= 0,
                     MU_Z_face_m * np.roll(f, 1, axis=1),
                     MU_Z_face_m * f)

    if bc_z == 'noflux':
        F_z_m[:, 0, :]  = 0.0   # bottom boundary: zero flux
        F_z_p[:, -1, :] = 0.0   # top boundary: zero flux

    div_z = -(F_z_p - F_z_m) / dz

    # ----------------------------------------------------
    # 3) Build "tilde" state after x,z explicit contributions
    # ----------------------------------------------------
    f_tilde = f + dt * (div_x + div_z)

    # ----------------------------------------------------
    # 4) Implicit phi-step with Chang–Cooper, periodic in phi
    #    Solve (f_new - f_tilde)/dt = L_phi(f_new)
    #    at each (i,k) line.
    # ----------------------------------------------------
    f_new = np.empty_like(f)

    for i in range(Nx):
        for k in range(Nz):
            f_line  = f_tilde[i, k, :]
            A_line  = MU_PHI[i, k, :]
            D_line  = D[i, k, :]

            f_new[i, k, :] = cc_phi_implicit_line(f_line, A_line, D_line, dt, dphi)

    # Enforce positivity (numerical noise)
    f_new = np.maximum(f_new, 0.0)

    return f_new


# ========== Time Integration ==========

def solve_fokker_planck_implicit_phi(f0, t_array, x_grid, z_grid, phi_grid,
                                     bc_x='open', bc_z='noflux', bc_phi='periodic',
                                     verbose=True, live_plot=False, plot_interval=50):
    """
    Time integration using semi-implicit solver:
      x,z explicit; phi implicit Chang–Cooper.
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
        fig.suptitle('Live Fokker-Planck Evolution (implicit phi)', fontsize=14)

    if verbose:
        print("Starting semi-implicit time integration (implicit in φ)...")

    for n in range(Nt - 1):
        dt = t_array[n+1] - t_array[n]
        t  = t_array[n]

        f = fokker_planck_step_implicit_phi(
            f, X, Z, PHI, t, dt, dx, dz, dphi,
            bc_x=bc_x, bc_z=bc_z, bc_phi=bc_phi
        )

        solution[n+1] = f

        if np.any(np.isnan(f)):
            print(f"NaN detected at step {n+1}!")
            break

        if verbose and ((n+1) % 100 == 0 or n == 0):
            total_prob = np.sum(f) * dx * dz * dphi
            print(f"Step {n+1}/{Nt-1}, t={t_array[n+1]:.4f}, ∫f={total_prob:.6f}, "
                  f"min={f.min():.3e}, max={f.max():.3e}")

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
    
    solution = solve_fokker_planck_implicit_phi(
        f0, t_array, x_grid, z_grid, phi_grid,
        bc_x='open', bc_z='noflux', bc_phi='periodic',
        live_plot=False
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