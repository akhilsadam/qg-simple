import torch

def int_sq(y, grid):
    Y = torch.sum(torch.abs(y[:, 0])**2) + 2*torch.sum(torch.abs(y[:, 1:])**2)
    n = grid.Lx * grid.Ly  # Use grid object for Lx and Ly
    return Y * n

# Generates initial conditions based on specified energy and wavenumber limits
def _init_randn(energy, wavenumbers, grid, spectral_derivative, seed=86, **kwargs):
    torch.manual_seed(seed)
    
    # Use spectral_derivative for kr, ky, and krsq
    K = torch.sqrt(spectral_derivative.krsq)  # Wavenumber of each point in frequency space
    k = spectral_derivative.kr.repeat(grid.Ny, 1)  # Ensure proper shape for k

    # Generate random complex field in spectral space
    qih = torch.randn(spectral_derivative.krsq.size(), dtype=torch.complex128).to(grid.device)
    
    # Apply wavenumber filters
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0  # Handle zero wavenumber 
    
    # Normalize initial condition energy
    E0 = energy
    Ei = 0.5 * (int_sq(spectral_derivative.kr * spectral_derivative.irsq * qih, grid) +
                int_sq(spectral_derivative.ky * spectral_derivative.irsq * qih, grid)) / (grid.Lx * grid.Ly)
    
    # Scale to the desired energy
    qih *= torch.sqrt(E0 / Ei)
    return qih

####################################################################################################

valid_ic = lambda _ic: isinstance(_ic, dict) and 'function' in _ic and _ic['function'] in ic_library

ic_library = {
    'randn': _init_randn,
}

solve_ic =  lambda _ic: ic_library[_ic['function']](**_ic) if valid_ic(_ic) else _ic