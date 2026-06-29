import torch
import numpy as np
from PIL import Image
import os

# TODO add subpixel rendering / anti-aliasing to non-SDF methods

def add_margin(pil_img, width, height, top, left, color):
    result = Image.new(pil_img.mode, (width, height), color)
    result.paste(pil_img, (left, top))
    return result

def sdf_raster(d, r, tolerance):
    # mask = torch.zeros_like(d)
    # mask[d < r] = 1  # Inside the circle
    # mask[torch.abs(d - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    mask = torch.clamp((r - d) / tolerance, -0.5, 0.5) + 0.5
    
    return mask

def circular(grid, derivative, # add state as first argument if time-dependent
             r, tolerance=1, invert=False,
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
    mask = sdf_raster(distance, r, tolerance * max(Lx/Nx, Ly/Ny))
    
    if invert:
        mask = 1 - mask
    
    return mask[None,:,:]  # Add batch dimension

def box(grid, derivative, # add state as first argument if time-dependent
             r, tolerance=1, invert=False,
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
    distance = torch.max(torch.abs(x[None,:] - x_center), torch.abs(y[:,None] - y_center)) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = sdf_raster(distance, r, tolerance * max(Lx/Nx, Ly/Ny))
    
    if invert:
        mask = 1 - mask
    
    return mask[None,:,:]  # Add batch dimension

def fpc(grid, derivative, # add state as first argument if time-dependent
             tolerance=1,
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
    mask = sdf_raster(distance, r, tolerance * max(Lx/Nx, Ly/Ny))

    return mask[None,:,:]  # Add batch dimension

def cape(grid, derivative, height=1/4, sigma=1/16, tolerance=1, pad=0.24, **kwargs):
    # Use grid object for domain size and number of grid points
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Nx/Ny, Nx, device = grid.device)[None, :]
    y = torch.flip(torch.linspace(0, 1, Ny, device = grid.device), (0,))[:, None]

    # Find the center of the domain
    x_center = 3 / 16
    y_center = pad + 1 / 8

    # vertical cape profile
    cape_profile = (y_center + height * torch.exp(-((x-x_center)/sigma)**2) - y)
    cape = (cape_profile > 0) \
        * (y > y_center - 1/8)

    mask = sdf_raster(-1 * cape, 0, tolerance * max(Lx/Nx, Ly/Ny))
    
    return mask[None,:,:]  # Add batch dimension

def im(grid, derivative, # add state as first argument if time-dependent
             mask='osk.png', th=0.5, blur=0.0, pad=0, pad_mode='lrtd',
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    if pad > 0:
        # pad image out with 0-padding
        pads = [0,0,0,0]
        if 'l' in pad_mode: # left-right swap
            pads[1] = pad
        if 'r' in pad_mode:
            pads[0] = pad
        if 't' in pad_mode:
            pads[2] = pad
        if 'd' in pad_mode:
            pads[3] = pad
    
    if isinstance(mask, str):
        # get current file path
        path = os.path.dirname(os.path.abspath(__file__))
        
        # check if mask is not an absolute path
        if not os.path.isabs(mask):
            mask = os.path.join(path, 'mask', mask)
        img = Image.open(mask)        
        
        # img = img.resize((Nx, Ny))
        # pad image if pad > 0
        if pad > 0:
            img = img.resize((Nx - pads[0] - pads[1], Ny - pads[2] - pads[3]))
            img = add_margin(img, Nx, Ny, pads[0], pads[1], 0)
        else:
            img = img.resize((Nx, Ny))
        
        img = img.convert('L')
        img = np.array(img).astype(np.float32)
        img /= np.max(img)
        img = torch.tensor(img)
        
    else:
        raise ValueError(f"Mask should be a string path to an image file, got {mask} instead. Not implemented yet.")
        
    
        
    mask = torch.where(img > th, 1.0, 0.0).to(grid.device, dtype=grid.ftype)

    if blur > 0.0:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask.cpu().numpy(), sigma=blur)
        mask = torch.tensor(mask).to(grid.device, dtype=grid.ftype)
    
    return mask[None,:,:]  # Add batch dimension


def nc(grid, derivative, # add state as first argument if time-dependent
             mask='riot_070725', clip=-200, pad=0.08, pad_mode='lrtd',
             **kwargs):
    
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny


    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(mask):
        mask_path = os.path.join(path, 'mask', f'{mask}.nc')
        npy_path = os.path.join(path, 'mask', f'{mask}.npy')
        png_path = os.path.join(path, 'mask', f'{mask}.png')
        
    if not os.path.exists(png_path):
        if os.path.exists(npy_path):
            data = np.load(npy_path)
        elif os.path.exists(mask_path):
            from netCDF4 import Dataset
            with Dataset(mask_path, 'r') as nc_file:
                data = nc_file.variables[nc_library[mask]]
                data = np.array(data)
                print(data.shape)
                print(np.min(data), np.max(data))
                np.save(npy_path, data)
        
        mask = torch.from_numpy(data > clip)
        Image.fromarray(mask.numpy()).save(png_path)
    
    return im(grid, derivative, mask=png_path, th=0.5, blur=0.0, pad=int(pad * Nx), pad_mode=pad_mode) # use im function to return mask


####################################################################################################

valid_mask = lambda _mask: hasattr(_mask, 'function') and _mask.function in mask_library

mask_library = {
    'fpc': fpc,
    'circular': circular,
    'box' : box,
    'image': im,
    'netCDF': nc,
}

nc_library = {
    'riot_070725': 'raw_bath', # /gdata/projects/dri_riot/Grids/2025/Jul07/socal0450m/grids_riot_sa0450m.nc; point conception, channel islands near Santa Barbara, CA
}
        

def solve_mask(_mask):
    if _mask is None or not isinstance(_mask, dict) or 'function' not in _mask or _mask['function'] not in mask_library:
        return _mask
    return lambda *args: mask_library[_mask['function']](*args, **_mask)

