import torch
import math

### Set up spectral derivatives (first and second derivatives)
class Derivative:
    def __init__(self, grid):
        self.grid = grid
        self.device = grid.device

        self._dx =  grid.Lx / (grid.Nx)
        self._dy =  grid.Ly / (grid.Ny)
        

        # Number of wavenumber components (half of real grid in x-direction)
        self.dk = int(grid.Nx / 2 + 1)

        ## Compute wavenumbers for first derivatives
        # Derivative in y
        self.ky = torch.reshape((torch.fft.fftfreq(grid.Ny, grid.Ly / (grid.Ny * 2 * torch.pi))), 
            (grid.Ny, 1)
        )[None,:,:].to(self.device) 
        
        # Derivative in x
        self.kx = torch.reshape((torch.fft.rfftfreq(grid.Nx, grid.Lx / (grid.Nx * 2 * torch.pi))), 
            (1, self.dk)
        )[None,:,:].to(self.device) # also kr (radial)
        
        self.dx = 1j * self.kx
        self.dy = 1j * self.ky

        # Squared wavenumbers (for second derivatives)
        self.krsq = self.kx**2 + self.ky**2  

        # Inverse squared wavenumbers (and handling zero division)
        self.irsq = 1.0/self.krsq
        self.irsq[:,0,0] = 0.0 #
        
        # Dealiasing wavenumber for stability and mask
        dealias_factor=1/3
        self.k_cut = math.sqrt(2) * (1 - dealias_factor) * min(self.ky.max(), self.kx.max())
        
        self.sqrt_krsq = torch.sqrt(self.krsq)
        self.alias_mask = (self.sqrt_krsq > self.k_cut)
        
    def dealias(self, y):
        """
        Apply dealiasing to the field based on the ratio (usually 1/3 rule).
        The field's high-frequency components are truncated.
        
        Args:
        - y: tensor in spectral space to apply dealiasing.
        - derivative: SpectralOperator instance to access ky, kr, and krsq.
        - dealias_factor: factor to apply the dealiasing (default is 1/3).
        
        Returns:
        - y: tensor with high frequencies removed.
        """
        # Apply dealiasing: set high-frequency components to zero
        y[self.alias_mask.expand_as(y)] = 0
        return y

        
    def to(self, device):
        """ Move spectral operator tensors to another device. """
        self.device = device
        self.dx = self.dx.to(device)
        self.dy = self.dy.to(device)
        self.kx = self.kx.to(device)
        self.ky = self.ky.to(device)
        self.krsq = self.krsq.to(device)
        self.irsq = self.irsq.to(device)

    def __repr__(self):
        return (f"Derivative: Nx={self.grid.Nx}, Ny={self.grid.Ny}, dk={self.dk}, "
                f"Lx={self.grid.Lx:.4f}, Ly={self.grid.Ly:.4f}, device={self.device}")
