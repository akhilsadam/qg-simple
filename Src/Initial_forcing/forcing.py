import torch
import numpy as np
from Operators.spectral_conversion import to_physical, to_spectral, dealias

# Generates forcing based on specified wavenumber and time effects

def wind_forcing(grid,spectral_derivative,wind_params,t):
    
    # Create a grid of coordinates (x, y)
    x = np.linspace(0, grid.Lx, grid.Nx)
    y = np.linspace(0, grid.Ly, grid.Ny)

    # Create meshgrid for x, y
    X, Y = np.meshgrid(x, y)
    
    w = wind_params.A * (np.cos(wind_params.B * X + wind_params.C * t)) + wind_params.D * (np.cos(wind_params.E * Y + wind_params.F * t))
    
    wh = to_spectral(w)
    
    return dealias(wh,spectral_derivative,1/3)
