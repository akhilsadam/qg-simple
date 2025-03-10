import torch
from solver.opt.basis import to_spectral, dealias

# Generates forcing based on specified wavenumber and time effects
def unscaled_cosine(t,grid,spectral_derivative,forcing_params):
    # grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None] # meshgrid for x, y
    
    w = forcing_params.A * (torch.cos(forcing_params.B * X + forcing_params.C * t)) \
        + forcing_params.D * (torch.cos(forcing_params.E * Y + forcing_params.F * t))
    
    wh = to_spectral(w)
    return dealias(wh,spectral_derivative,1/3)

####################################################################################################

valid_fc = lambda _fc: isinstance(_fc, dict) and 'function' in _fc and _fc['function'] in fc_library

fc_library = {
    'unscaled_cosine': unscaled_cosine,
}

solve_forcing =  lambda _fc: fc_library[_fc['function']](**_fc) if valid_fc(_fc) else _fc