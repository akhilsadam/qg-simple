import torch

def int_sq(y, grid):
    Y = torch.sum(torch.abs(y[:, 0])**2) + 2*torch.sum(torch.abs(y[:, 1:])**2)
    n = grid.Lx * grid.Ly  # Use grid object for Lx and Ly
    return Y * n

# Generates initial conditions based on specified energy and wavenumber limits
def _init_randn(grid, derivative,
                energy=0.0, wavenumbers=[3.0, 5.0], n_batch = 1,
                seed=86, persistent=True,
                **kwargs):
    
    if persistent and 'ic_' in globals():
        return globals()['ic_'].detach().clone()
    
    torch.manual_seed(seed)
    
    # Use derivative for kr, ky, and krsq
    K = torch.sqrt(derivative.krsq).repeat(n_batch, 1, 1)  # Wavenumber of each point in frequency space
    k = derivative.kx.repeat(n_batch, grid.Ny, 1)                # Ensure proper shape for k

    # Generate random complex field in spectral space
    qih = torch.randn(k.size(), dtype=torch.complex128).to(grid.device)
    
    # Apply wavenumber filters
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0  # Handle zero wavenumber 
    
    # Normalize initial condition energy
    E0 = energy
    Ei = 0.5 * (int_sq(derivative.kx * derivative.irsq * qih, grid) +
                int_sq(derivative.ky * derivative.irsq * qih, grid)) / (grid.Lx * grid.Ly)
    
    # Scale to the desired energy
    qih *= torch.sqrt(E0 / Ei)
    
    # Store the initial condition for persistent use
    if persistent:
        globals()['ic_'] = qih.detach().clone()
    
    # print("Initial condition energy:", E0)
    # print(torch.max((qih).abs()), torch.min((qih).abs()))
    
    return qih

####################################################################################################

valid_ic = lambda _ic: hasattr(_ic, 'function') and _ic.function in ic_library

ic_library = {
    'randn': _init_randn,
}

def solve_ic(_ic):
    if _ic is None or not isinstance(_ic, dict) or 'function' not in _ic or _ic['function'] not in ic_library:
        return _ic
    return lambda *args: ic_library[_ic['function']](*args, **_ic)