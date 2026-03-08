import torch
import torch.nn.functional as F
from qg.solver.opt.basis import to_physical, to_spectral

# Generates forcing based on specified wavenumber and time effects
def unscaled_cosine(state, grid, derivative, 
                    A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0,
                    **kwargs):
    # grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None] # meshgrid for x, y
    
    w = A * (torch.cos(B * X + C * state.t)) \
        + D * (torch.cos(E * Y + F * state.t)) [None,:,:]
    
    wh = to_spectral(w)
    return wh

def local_cosines(state, grid, derivative, 
                    A=[2.0], kx=[2.0], ky=[2.0], cx=[0.5], cy=[0.5], w=0.0, sigma=[0.1], **kwargs):
    # grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None] # meshgrid for x, y
    
    omega = 0.0
    for Ai, kxi, kyi, cxi, cyi, sigmai in zip(A, kx, ky, cx, cy, sigma):
        # Gaussian envelope centered at (cx * Lx, cy * Ly) with width sigma
        if sigmai > 0:
            gsn = torch.exp(-((X - cxi * grid.Lx)**2 + (Y - cyi * grid.Ly)**2) / (2 * (sigmai * grid.Lx)**2)) / (2 * torch.pi * (sigmai * grid.Lx)**2) # Gaussian envelope
        else:
            gsn = 1.0
        omega = omega + gsn * Ai * (torch.cos(kxi * X + w * state.t) \
            + torch.cos(kyi * Y + w * state.t)) [None,:,:]
    
    wh = to_spectral(omega)
    return wh

####################################################################################################

valid_fc = lambda _fc: hasattr(_fc, 'function') and _fc.function in fc_library

fc_library = {
    'unscaled_cosine': unscaled_cosine,
    'local_cosines': local_cosines,
}

def solve_forcing(_fc):
    if _fc is None or not isinstance(_fc, dict) or 'function' not in _fc or _fc['function'] not in fc_library:
        return _fc
    return lambda *args: fc_library[_fc['function']](*args, **_fc)