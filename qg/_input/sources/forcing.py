import torch
from qg.solver.opt.basis import to_spectral, dealias

# Generates forcing based on specified wavenumber and time effects
def unscaled_cosine(state, grid, spectral_derivative, 
                    A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0,
                    **kwargs):
    # grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None] # meshgrid for x, y
    
    w = A * (torch.cos(B * X + C * state.t)) \
        + D * (torch.cos(E * Y + F * state.t))
    
    wh = to_spectral(w)
    return dealias(wh,spectral_derivative,1/3)

####################################################################################################

valid_fc = lambda _fc: hasattr(_fc, 'function') and _fc.function in fc_library

fc_library = {
    'unscaled_cosine': unscaled_cosine,
}

solve_forcing = lambda _fc: (lambda *args: fc_library[_fc.function](*args, **_fc.__dict__)) if valid_fc(_fc) else _fc