import numpy as np

# ========== Model Parameters ==========
alpha = 0.1
v0 = 1.0
sigma = 0.5
beta = 0.0
vg = 0.5
lamda = 0.3
psi = 1.0

# ========== Drift and Diffusion Functions ==========

def mu_x(vec, t): 
    x, z, phi = vec
    return alpha*np.exp(z)*np.cos(x-t) + v0*np.sin(phi) + sigma*(beta+z)

def mu_z(vec, t):
    x, z, phi = vec
    return alpha*np.exp(z)*np.sin(x-t) + v0*np.cos(phi) - vg

def mu_phi(vec, t):
    x, z, phi = vec
    return (lamda*alpha*np.exp(z)*np.cos(x-t+2*phi) - 
            1/(2*psi)*np.sin(phi) + 
            sigma/2*(1+lamda*np.cos(2*phi)))

def var_fun(vec, t):
    """Noise intensity (standard deviation)"""
    return 0.3

# ========== Helper Functions ==========

def delta_cc(Pe):
    """Chang-Cooper delta function"""
    if abs(Pe) < 1e-12:
        return 0.5
    return 1.0/Pe - 1.0/(np.exp(Pe) - 1.0)

def get_neighbor_index(i, N, bc):
    """Get neighbor index with boundary condition handling"""
    if bc == 'periodic':
        return i % N
    elif bc in ['open', 'absorbing', 'zero_gradient', 'noflux']:
        if i < 0 or i >= N:
            return None
        return i
    return max(0, min(i, N-1))

# ========== Flux Computation Functions ==========

def compute_upwind_flux(f, mu_face, i_curr, i_next, j, k, at_boundary, bc):
    """
    Compute upwind flux for advection
    Returns flux at face between i_curr and i_next
    """
    if i_next is None:  # At boundary
        if bc == 'noflux':
            return 0.0
        elif bc in ['open', 'absorbing']:
            # Only allow outflow
            return max(0, mu_face) * f[i_curr, j, k]
        elif bc == 'zero_gradient':
            return mu_face * f[i_curr, j, k]
        else:
            return 0.0
    
    # Interior: standard upwind
    if mu_face >= 0:
        return mu_face * f[i_curr, j, k]
    else:
        return mu_face * f[i_next, j, k]

def compute_x_fluxes(f, x_grid, z, phi, t, i, j, k, im, ip, bc_x):
    """Compute x-direction fluxes at faces i±1/2"""
    vec = np.array([x_grid[i], z, phi])
    
    # Face i+1/2
    if ip is not None:
        vec_right = np.array([x_grid[ip], z, phi])
        mu_face_p = 0.5 * (mu_x(vec, t) + mu_x(vec_right, t))
    else:
        mu_face_p = mu_x(vec, t)
    
    F_x_p = compute_upwind_flux(f, mu_face_p, i, ip, j, k, (ip is None), bc_x)
    
    # Face i-1/2
    if im is not None:
        vec_left = np.array([x_grid[im], z, phi])
        mu_face_m = 0.5 * (mu_x(vec_left, t) + mu_x(vec, t))
    else:
        mu_face_m = mu_x(vec, t)
    
    # For left boundary, flip logic
    if im is None:
        if bc_x in ['open', 'absorbing']:
            F_x_m = min(0, mu_face_m) * f[i, j, k]
        elif bc_x == 'zero_gradient':
            F_x_m = mu_face_m * f[i, j, k]
        else:
            F_x_m = 0.0
    else:
        if mu_face_m >= 0:
            F_x_m = mu_face_m * f[im, j, k]
        else:
            F_x_m = mu_face_m * f[i, j, k]
    
    return F_x_p, F_x_m

def compute_z_fluxes(f, z_grid, x, phi, t, i, j, k, jm, jp, bc_z):
    """Compute z-direction fluxes at faces j±1/2"""
    vec = np.array([x, z_grid[j], phi])
    
    # Face j+1/2
    if jp is not None:
        vec_right = np.array([x, z_grid[jp], phi])
        mu_face_p = 0.5 * (mu_z(vec, t) + mu_z(vec_right, t))
        if mu_face_p >= 0:
            F_z_p = mu_face_p * f[i, j, k]
        else:
            F_z_p = mu_face_p * f[i, jp, k]
    else:
        F_z_p = 0.0  # No-flux boundary
    
    # Face j-1/2
    if jm is not None:
        vec_left = np.array([x, z_grid[jm], phi])
        mu_face_m = 0.5 * (mu_z(vec_left, t) + mu_z(vec, t))
        if mu_face_m >= 0:
            F_z_m = mu_face_m * f[i, jm, k]
        else:
            F_z_m = mu_face_m * f[i, j, k]
    else:
        F_z_m = 0.0  # No-flux boundary
    
    return F_z_p, F_z_m

def compute_chang_cooper_flux(f, A, D, dphi, i, j, k, k_neighbor):
    """Compute Chang-Cooper flux for drift-diffusion"""
    if k_neighbor is None:
        return 0.0
    
    if abs(D) < 1e-14:
        # Pure advection
        if A >= 0:
            return A * f[i, j, k]
        else:
            return A * f[i, j, k_neighbor]
    
    # Drift-diffusion
    Pe = A * dphi / D
    delta = delta_cc(Pe)
    
    flux = (A * (1.0 - delta) * f[i, j, k_neighbor] + 
            A * delta * f[i, j, k] - 
            D * (f[i, j, k_neighbor] - f[i, j, k]) / dphi)
    
    return flux

def compute_phi_fluxes(f, phi_grid, x, z, t, i, j, k, km, kp, dphi):
    """Compute phi-direction Chang-Cooper fluxes at faces k±1/2"""
    vec = np.array([x, z, phi_grid[k]])
    
    # Face k+1/2
    if kp is not None:
        vec_right = np.array([x, z, phi_grid[kp]])
        A_p = 0.5 * (mu_phi(vec, t) + mu_phi(vec_right, t))
        D_p = 0.25 * (var_fun(vec, t)**2 + var_fun(vec_right, t)**2)
        J_phi_p = compute_chang_cooper_flux(f, A_p, D_p, dphi, i, j, k, kp)
    else:
        J_phi_p = 0.0
    
    # Face k-1/2
    if km is not None:
        vec_left = np.array([x, z, phi_grid[km]])
        A_m = 0.5 * (mu_phi(vec_left, t) + mu_phi(vec, t))
        D_m = 0.25 * (var_fun(vec_left, t)**2 + var_fun(vec, t)**2)
        J_phi_m = compute_chang_cooper_flux(f, A_m, D_m, dphi, i, j, km, k)
    else:
        J_phi_m = 0.0
    
    return J_phi_p, J_phi_m

# ========== Main Time-Stepping Function ==========

def fokker_planck_step(f, x_grid, z_grid, phi_grid, t, dt,
                       bc_x='open', bc_z='noflux', bc_phi='periodic'):
    """
    One explicit Euler step for 3D Fokker-Planck equation
    ∂f/∂t = -∂(μ_x f)/∂x - ∂(μ_z f)/∂z - ∂(μ_φ f)/∂φ + ∂(D ∂f/∂φ)/∂φ
    """
    Nx, Nz, Nphi = f.shape
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    f_new = np.zeros_like(f)
    
    for i in range(Nx):
        for j in range(Nz):
            for k in range(Nphi):
                x, z, phi = x_grid[i], z_grid[j], phi_grid[k]
                
                # Get neighbor indices
                ip = get_neighbor_index(i+1, Nx, bc_x)
                im = get_neighbor_index(i-1, Nx, bc_x)
                jp = get_neighbor_index(j+1, Nz, bc_z)
                jm = get_neighbor_index(j-1, Nz, bc_z)
                kp = get_neighbor_index(k+1, Nphi, bc_phi)
                km = get_neighbor_index(k-1, Nphi, bc_phi)
                
                # Compute fluxes
                F_x_p, F_x_m = compute_x_fluxes(f, x_grid, z, phi, t, i, j, k, im, ip, bc_x)
                F_z_p, F_z_m = compute_z_fluxes(f, z_grid, x, phi, t, i, j, k, jm, jp, bc_z)
                J_phi_p, J_phi_m = compute_phi_fluxes(f, phi_grid, x, z, t, i, j, k, km, kp, dphi)
                
                # Compute divergences
                div_x = -(F_x_p - F_x_m) / dx
                div_z = -(F_z_p - F_z_m) / dz
                div_phi = -(J_phi_p - J_phi_m) / dphi
                
                # Update
                dfdt = div_x + div_z + div_phi
                f_new[i, j, k] = f[i, j, k] + dt * dfdt
    
    return f_new

# ========== Time Integration ==========

def solve_fokker_planck(f0, t_array, x_grid, z_grid, phi_grid,
                        bc_x='open', bc_z='noflux', bc_phi='periodic',
                        verbose=True):
    """Integrate Fokker-Planck equation over time"""
    Nt = len(t_array)
    Nx, Nz, Nphi = f0.shape
    
    solution = np.zeros((Nt, Nx, Nz, Nphi))
    solution[0] = f0.copy()
    
    f = f0.copy()
    
    if verbose:
        print("Starting time integration...")
    
    for n in range(Nt - 1):
        dt = t_array[n+1] - t_array[n]
        t = t_array[n]
        
        f = fokker_planck_step(f, x_grid, z_grid, phi_grid, t, dt,
                              bc_x, bc_z, bc_phi)
        
        solution[n+1] = f
        
        # Check for NaN
        if np.any(np.isnan(f)):
            print(f"NaN detected at step {n+1}!")
            break
        
        # Monitor progress
        if verbose and ((n+1) % 100 == 0 or n == 0):
            dx = x_grid[1] - x_grid[0]
            dz = z_grid[1] - z_grid[0]
            dphi = phi_grid[1] - phi_grid[0]
            total_prob = np.sum(f) * dx * dz * dphi
            min_val = np.min(f)
            max_val = np.max(f)
            print(f"Step {n+1}/{Nt-1}, t={t_array[n+1]:.4f}, "
                  f"∫f={total_prob:.6f}, min={min_val:.3e}, max={max_val:.3e}")
    
    return solution

def check_cfl_condition(x_grid, z_grid, phi_grid, dt, t=0):
    """Check CFL stability condition"""
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    D_max = 0.5 * var_fun(np.array([0, 0, 0]), t)**2
    
    if D_max > 0:
        dt_max_x = dx**2 / (2 * D_max)
        dt_max_z = dz**2 / (2 * D_max)
        dt_max_phi = dphi**2 / (2 * D_max)
        dt_max = min(dt_max_x, dt_max_z, dt_max_phi)
        
        print(f"CFL check: dt={dt:.4e}, dt_max={dt_max:.4e}")
        if dt > 0.5 * dt_max:
            print(f"⚠️  Warning: dt may be too large for stability!")
        else:
            print(f"✓ CFL condition satisfied")
    
    return dt

# ========== Example Usage ==========

if __name__ == "__main__":
    # Grid setup
    Nx, Nz, Nphi = 30, 30, 50
    x_grid = np.linspace(-20, 20, Nx)
    z_grid = np.linspace(-2, 2, Nz)
    phi_grid = np.linspace(0, 2*np.pi, Nphi)
    
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]
    dphi = phi_grid[1] - phi_grid[0]
    
    print(f"Grid: {Nx}×{Nz}×{Nphi}")
    print(f"Resolution: dx={dx:.3f}, dz={dz:.3f}, dφ={dphi:.3f}")
    
    # Initial condition: localized Gaussian
    X, Z, PHI = np.meshgrid(x_grid, z_grid, phi_grid, indexing='ij')
    x0, z0, phi0 = 0.0, 0.0, np.pi
    sigma_init = 0.5
    
    f0 = np.exp(-((X-x0)**2 + (Z-z0)**2 + (PHI-phi0)**2)/(2*sigma_init**2))
    f0 = f0 / (np.sum(f0) * dx * dz * dphi)
    
    print(f"Initial ∫f = {np.sum(f0)*dx*dz*dphi:.6f}\n")
    
    # Time setup
    t_final = 2.0
    dt = 0.001
    Nt = int(t_final / dt) + 1
    t_array = np.linspace(0, t_final, Nt)
    
    check_cfl_condition(x_grid, z_grid, phi_grid, dt)
    print()
    
    # Solve
    solution = solve_fokker_planck(f0, t_array, x_grid, z_grid, phi_grid,
                                   bc_x='open', bc_z='noflux', bc_phi='periodic')
    
    print(f"\n✓ Done! Solution shape: {solution.shape}")
    print(f"Final ∫f = {np.sum(solution[-1])*dx*dz*dphi:.6f}")