import math, inspect

# from schema import Schema, And, Use, Optional, Or

from mura.deploy.util import crupdate

from _input.sources.ic import solve_ic
from _input.sources.forcing import solve_forcing
from _input.mask.mask import solve_mask


# param_schema = Schema({
#     'grid': {
#         'Nx': int,
#         'Ny': int,
#         'Lx': float,
#         'Ly': float,
#     },
#     'time': {
#         'dt': float,
#         'T': float,
#         'save_rate': int,
#     },
#     'pde': {
#         'mu': float,
#         'nu': float,
#         'B': float,
#         'nv': int,
#         'penalty_coeff': float,
#     },
#     Optional('ic'): object,
#     Optional('forcing'): object,
#     Optional('mask'): object,
#     Optional(str): object,
# })
    
class grid:
    Nx = 512
    Ny = 512
    Lx = 2 * math.pi
    Ly = 2 * math.pi
    
class time:
    dt = 1e-3
    T = 100001 * dt
    save_rate = 500        # (Frequency of .np saves and plots)
    
class pde:
    mu = 0.0               # (Linear drag)
    nu = 1.025e-5          # (Viscosity coefficient)
    B = 2.195e2            # (Beta plane)
    nv = 1                 # (Hyperviscous order)
    penalty_coeff = 1.25   # (Brinkman penalty parameter)
    
class ic:
    energy = 0.01
    wavenumbers = [10.0, 32.0]
    seed= 86   

class forcing:
    function = 'unscaled_cosine'
    A=4
    B=4
    C=0
    D=1
    E=4
    F=0
            
class mask:
    function = 'circular'
    r = math.pi/4
    tol = 1e-3
    
class config:
    project_name = 'qg',
    cluster_name = 'mseas.mit.edu',

class validate():    
    def __init__(self, _params):
        
        self.grid = grid
        self.time = time
        self.pde = pde
        self.ic = ic
        self.forcing = forcing
        self.mask = mask
        
        crupdate(self, _params)
        
        self.ic = solve_ic(self.ic)
        self.forcing = solve_forcing(self.forcing)
        self.mask = solve_mask(self.mask)

