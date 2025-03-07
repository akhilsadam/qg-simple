import torch
import math
import numpy as np


class Grid:
    def __init__(self, Lx=2*math.pi, Ly=2*math.pi, Nx=512, Ny=512, device='cuda'):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.device = device

        self.size=self.Nx*self.Ny
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = torch.arange(-self.Lx/2, self.Lx/2, self.dx, device=self.device)
        self.y = torch.arange(-self.Ly/2, self.Ly/2, self.dy, device=self.device)
        
    def to(self, device):
        """ Move grid tensors to another device. """
        self.device = device
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __repr__(self):
        # print(grid)
        return (f"Grid(Lx={self.Lx}, Ly={self.Ly}, Nx={self.Nx}, Ny={self.Ny}, "
                f"dx={self.dx:.4f}, dy={self.dy:.4f}, device={self.device})")