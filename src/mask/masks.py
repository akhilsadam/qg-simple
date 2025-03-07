import torch
import numpy as np


def create_circular_mask(grid, r, tolerance=1e-3):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    # Create meshgrid for x, y
    X, Y = np.meshgrid(x, y)

    # Find the center of the domain
    x_center = Lx / 2
    y_center = Ly / 2

    # Compute the distance of each point from the center
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

    # Create the mask: inside the circle (distance < r) is 1
    mask = np.zeros_like(distance, dtype=np.float32)
    mask[distance < r] = 1  # Inside the circle
    mask[np.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    return torch.tensor(mask).to(grid.device)  # Move to the correct device (same as grid)