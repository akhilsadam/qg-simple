import math

# from schema import Schema, And, Use, Optional, Or

# from mura.deploy.util import crupdate

from mura.schema import design, field, MISSING

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
    
@design
class grid:
    Nx : int = 512
    Ny : int = 512
    Lx : float = 2 * math.pi
    Ly : float = 2 * math.pi
    precision : str = 'float32'   # float32 or float64
    
@design
class time:
    dt : float = 1e-3
    T : int = 200
    save_rate : int = 1000         # Frequency of .np saves and plots
    
@design
class pde:
    mu : float = 0.0                # Linear drag
    nu : float = 1.025e-5           # Viscosity coefficient
    B : float = 0.0                 # Beta plane
    nv : float = 1                  # Hyperviscous order
    penalty : float = 0.0           # Brinkman penalty parameter
    friction : float = None         # friction coefficient - between no-slip and free-slip
    rossby_radius : float = None    # Rossby radius for vortex stretching; still incorrect
    
@design
class ic:
    function : float = 'randn'
    energy : float = 0.01
    wavenumbers : list = field(default_factory = lambda: [10.0, 32.0])

@design
class forcing:
    function : str = 'unscaled_cosine'
    A : float = 0.25
    B : float = 4
    C : float = 0
    D : float = 0.25
    E : float = 4
    F : float = 0
          
@design  
class mask:
    function : str = 'circular'
    r : float = math.pi/4
    tol : float = 1e-3
    
@design
class LoggingConfig:
    project: str = "qg"
    task_name: str = MISSING # Must be provided
    run_name: str = MISSING # Must be provided
    run_id: str = 'auto' # autorun will automatically generate a run_id
    version: list = field(default_factory = lambda: [0, 0, 0])  # Major, Minor, Patch
    cluster_name = 'mseas.mit.edu'
    base_path: str = "./run"
    run_path: str = ''
    version_file: str = "version.yml"
    notes: str = ""
    
@design
class config:
    seed : int = 42
    logging : LoggingConfig = LoggingConfig() # mandatory
    grid : grid = grid()
    time : time = time()
    pde : pde = pde()
    ic : ic = ic()
    forcing: forcing = forcing()
    mask : mask = mask()

@design
class gconfig:
    checks = ['qg/solver']

class validate():    
    def __init__(self, param):
        
        self.runner = 'run.py'
        self.grid = param.grid
        self.time = param.time
        self.pde = param.pde
        self.ic = param.ic
        self.forcing = param.forcing
        self.mask = param.mask
        self.bc = None # TBD
       
        self.ic.seed = param.seed 

        
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
