import torch
import numpy as np
from PIL import Image
import os

def circular(grid, derivative, # add state as first argument if time-dependent
             r, tolerance=1e-3,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 2
    y_center = Ly / 2

    # Compute the distance of each point from the center
    distance = torch.sqrt((x[None,:] - x_center)**2 + (y[:,None] - y_center)**2) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = torch.zeros_like(distance)
    mask[distance < r] = 1  # Inside the circle
    mask[torch.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    return mask[None,:,:]  # Add batch dimension


def fpc(grid, derivative, # add state as first argument if time-dependent
             tolerance=1e-3,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 8
    y_center = Ly / 2
    r = Lx / 16  # Radius of the circle

    # Compute the distance of each point from the center
    distance = torch.sqrt((x[None,:] - x_center)**2 + (y[:,None] - y_center)**2) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = torch.zeros_like(distance)
    mask[distance < r] = 1  # Inside the circle
    mask[torch.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    return mask[None,:,:]  # Add batch dimension



def im(grid, derivative, # add state as first argument if time-dependent
             mask='osk.png', th=0.5, blur=0.0,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # open image as np array
    
    # get current file path
    path = os.path.dirname(os.path.abspath(__file__))
    
    # check if mask is not an absolute path
    if not os.path.isabs(mask):
        mask = os.path.join(path, 'mask', mask)
    
    
    img = Image.open(mask)
    img = img.resize((Nx, Ny))
    img = img.convert('L')
    img = np.array(img).astype(np.float32)
    img /= np.max(img)
    img = torch.tensor(img)
    mask = torch.where(img > th, 1.0, 0.0).to(grid.device, dtype=grid.ftype)

    if blur > 0.0:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask.cpu().numpy(), sigma=blur)
        mask = torch.tensor(mask).to(grid.device, dtype=grid.ftype)
    
    return mask[None,:,:]  # Add batch dimension


####################################################################################################

valid_mask = lambda _mask: hasattr(_mask, 'function') and _mask.function in mask_library

mask_library = {
    'fpc': fpc,
    'circular': circular,
    'image': im,
}

solve_mask = lambda _mask: (lambda *args: mask_library[_mask.function](*args, **_mask.__dict__)) if valid_mask(_mask) else _mask