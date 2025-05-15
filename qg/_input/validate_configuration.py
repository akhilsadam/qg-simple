import math

# from schema import Schema, And, Use, Optional, Or

from mura.deploy.util import crupdate



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
    precision = 'float32'   # float32 or float64
    
class time:
    dt = 1e-3
    T = 100
    save_rate = 500         # Frequency of .np saves and plots
    
class pde:
    mu = 0.0                # Linear drag
    nu = 1.025e-5           # Viscosity coefficient
    B = 5.0                 # Beta plane
    nv = 1                  # Hyperviscous order
    penalty = 1.25          # Brinkman penalty parameter
    friction = None         # friction coefficient - between no-slip and free-slip
    rossby_radius = None    # Rossby radius for vortex stretching
    
class ic:
    function = 'randn'
    energy = 0.01
    wavenumbers = [10.0, 32.0]
    seed= 86   

class forcing:
    function = 'unscaled_cosine'
    A=0.25
    B=4
    C=0
    D=0.25
    E=4
    F=0
            
class mask:
    function = 'circular'
    r = math.pi/4
    tol = 1e-3
    
class config:
    project_name = 'qg'
    cluster_name = 'mseas.mit.edu'
    
class gconfig:
    checks = ['qg/solver']

class validate():    
    def __init__(self, _params):
        
        self.runner = 'run.py'
        self.grid = grid
        self.time = time
        self.pde = pde
        self.ic = ic
        self.forcing = forcing
        self.mask = mask
        
        crupdate(self, _params)
        
    def solve(self):
        # imports here to avoid import on load
        from qg._input.sources.ic import solve_ic
        from qg._input.sources.bc import solve_bc
        from qg._input.sources.forcing import solve_forcing
        from qg._input.mask.mask import solve_mask
        
        self.ic = solve_ic(self.ic)
        self.bc = solve_bc(self.bc)
        self.mask = solve_mask(self.mask)
        self.forcing = solve_forcing(self.forcing)
        
        return self
