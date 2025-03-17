import torch

### Set up spectral derivatives (first and second derivatives)
class Derivative:
    def __init__(self, grid):
        self.grid = grid
        self.device = grid.device

        # Number of wavenumber components (half of real grid in x-direction)
        self.dk = int(grid.Nx / 2 + 1)

        ## Compute wavenumbers for first derivatives
        # Derivative in y
        self.ky = torch.reshape((torch.fft.fftfreq(grid.Ny, grid.Ly / (grid.Ny * 2 * torch.pi))), 
            (grid.Ny, 1)
        )[None,:,:].to(self.device) 
        
        # Derivative in x
        self.kr = torch.reshape((torch.fft.rfftfreq(grid.Nx, grid.Lx / (grid.Nx * 2 * torch.pi))), 
            (1, self.dk)
        )[None,:,:].to(self.device)

        # Squared wavenumbers (for second derivatives)
        self.krsq = self.kr**2 + self.ky**2  

        # Inverse squared wavenumbers (and handling zero division)
        self.irsq = 1.0/self.krsq
        self.irsq[:,0,0] = 0.0 #

    def to(self, device):
        """ Move spectral operator tensors to another device. """
        self.device = device
        self.ky = self.ky.to(device)
        self.kr = self.kr.to(device)
        self.krsq = self.krsq.to(device)
        self.irsq = self.irsq.to(device)

    def __repr__(self):
        return (f"Derivative: Nx={self.grid.Nx}, Ny={self.grid.Ny}, dk={self.dk}, "
                f"Lx={self.grid.Lx:.4f}, Ly={self.grid.Ly:.4f}, device={self.device}")
