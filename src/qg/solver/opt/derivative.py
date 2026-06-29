import torch
import math

from qg.solver.opt.basis import to_physical, to_spectral

### Set up spectral derivatives (first and second derivatives)
class Derivative:
    def __init__(self, grid):
        self.grid = grid
        self.device = grid.device
        
        # normal model
        self.mkx = grid.kx
        self.mky = grid.ky
        
        # Zeitlin sine?
        # _dx = torch.pi / grid.Nx
        # _dy = torch.pi / grid.Ny
        # self.mkx = torch.sin(grid.kx * _dx) / _dx
        # self.mky = torch.sin(grid.ky * _dy) / _dy
        
        self.dx = 1j * self.mkx
        self.dy = 1j * self.mky

        # laplacian
        self.mksq = self.mkx**2 + self.mky**2  
        self.laplacian = - self.mksq
        self.inv_laplacian = 1.0/self.laplacian
        self.inv_laplacian[:,0,0] = 0.0 # 0/0 = 0 
        
        # dealiasing for stability and mask
        dealias_factor=1/3
        self.k_cut = math.sqrt(2) * (1 - dealias_factor) * min(self.mky.max(), self.mkx.max())
        self.alias_mask = (torch.sqrt(self.mksq) > self.k_cut)
        
        # gaussian smoothing
        sigma = 2 # in pixels
        self.blur = torch.exp(-0.5 * (sigma * grid.dx) ** 2 * self.mksq)
        
    def dealias(self, y):
        """
        Apply dealiasing to the field based on the ratio (usually 1/3 rule).
        The field's high-frequency components are truncated.
        """
        if isinstance(y, (tuple, list)):
            return tuple(self.dealias(v) for v in y)
        # Apply dealiasing: set high-frequency components to zero
        y[self.alias_mask.expand_as(y)] = 0
        return y

    def grad(self, scalar_h):
        return self.dx * scalar_h, self.dy * scalar_h

    def div(self, vector_h):
        vx_h, vy_h = vector_h
        return self.dx * vx_h + self.dy * vy_h

    def curl(self, vector_h):
        vx_h, vy_h = vector_h
        return self.dx * vy_h - self.dy * vx_h

    def inner(self, a_h, b_h):
        ax_h, ay_h = a_h
        bx_h, by_h = b_h
        return to_spectral(
            to_physical(ax_h) * to_physical(bx_h)
            + to_physical(ay_h) * to_physical(by_h)
        )

        
    def to(self, device):
        self.device = device
        self.dx = self.dx.to(device)
        self.dy = self.dy.to(device)
        self.laplacian = self.laplacian.to(device)
        self.inv_laplacian = self.inv_laplacian.to(device)
        self.alias_mask = self.alias_mask.to(device)
        self.blur = self.blur.to(device)
        return self

    def __repr__(self):
        return (f"Derivative: Nx={self.grid.Nx}, Ny={self.grid.Ny}, dk={self.dk}, "
                f"Lx={self.grid.Lx:.4f}, Ly={self.grid.Ly:.4f}, device={self.device}")

if __name__ == "__main__":
    from qg.solver.grid.cartesian import CartesianGrid
    grid = CartesianGrid(Nx=64, Ny=64)
    derivative = Derivative(grid)
    print(derivative.mkx)