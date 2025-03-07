import torch
import numpy as np
from Operators.spectral_conversion import to_physical, to_spectral, dealias

# Generates forcing based on specified wavenumber and time effects

def cos_forcing(grid,spectral_derivative,forcing_params,t):
    
    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)

    # Create meshgrid for x, y
    X, Y = x[None,:],y[:,None]
    
    w =  torch.tensor(forcing_params.A * (torch.cos(forcing_params.B * X + forcing_params.C * t)) + forcing_params.D * (torch.cos(forcing_params.E * Y + forcing_params.F * t)))
    
    wh = to_spectral(w)
    
    return dealias(wh,spectral_derivative,1/3)
