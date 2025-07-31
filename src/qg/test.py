from qg import config, autorun

def test_decaying_qg_turbulence():
    _config = config()
    _config.logging.task_name = "test_decaying_qg_turbulence"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 512
    _config.grid.Ny = 512
    _config.time.T = 60
    _config.ic.n_batch = 20
    autorun(_config)

def test_fpc():
    _config = config()
    _config.logging.task_name = "test_fpc"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 256
    _config.grid.Ny = 256
    _config.pde.nu = 5e-3 # RE 200
    _config.pde.penalty = 4.0 # Brinkman penalty parameter for cylinder
    _config.ic.wavenumbers = [1, 3] # param.data_wavenumbers
    _config.ic.energy = 0.0005
    _config.mask.function = 'fpc'
    _config.bc.function = 'const-outlet-r'
    _config.time.dt = 5e-4 # param.dt
    _config.time.save_rate = 500 # param.sim_steps
    _config.time.T = 120
    _config.ic.n_batch = 1
    _config.fps = 4
    autorun(_config)


def test_ideal_cape_high_re():
    _config = config()
    _config.logging.task_name = "test_ideal_cape_high_re"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 256
    _config.grid.Ny = 256
    _config.pde.nu = 5e-4 # RE 2000
    _config.pde.penalty = 4.0 # Brinkman penalty parameter for cylinder
    _config.ic.wavenumbers = [1, 3] # param.data_wavenumbers
    _config.ic.energy = 0.0005
    _config.mask.function = 'cape'
    _config.bc.function = 'const-outlet-rtd'
    _config.bc.width = 0.08 # Width of the sponge region
    _config.time.dt = 1e-4 # param.dt
    _config.time.save_rate = 500 # param.sim_steps
    _config.time.T = 12
    _config.ic.n_batch = 1
    _config.fps = 4
    autorun(_config)
    
def test_ideal_cape_low_re():
    _config = config()
    _config.logging.task_name = "test_ideal_cape_low_re"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 256
    _config.grid.Ny = 256
    _config.pde.nu = 5e-3 # RE 200
    _config.pde.penalty = 4.0 # Brinkman penalty parameter for cylinder
    _config.ic.wavenumbers = [1, 3] # param.data_wavenumbers
    _config.ic.energy = 0.0005
    _config.mask.function = 'cape'
    _config.mask.height = 1/8 # Height of the cape
    _config.bc.function = 'const-outlet-rtd'
    _config.bc.width = 0.08 # Width of the sponge region
    _config.time.dt = 1e-4 # param.dt
    _config.time.save_rate = 500 # param.sim_steps
    _config.time.T = 12
    _config.ic.n_batch = 1
    _config.fps = 4
    autorun(_config)


def test_cape():
    _config = config()
    _config.logging.task_name = "test_cape"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 256
    _config.grid.Ny = 256
    _config.pde.nu = 5e-4 # RE 200
    _config.pde.penalty = 4.0 # Brinkman penalty parameter for cylinder
    _config.ic.wavenumbers = [1, 3] # param.data_wavenumbers
    _config.ic.energy = 0.0005
    _config.mask.function = 'image'
    _config.mask.mask = 'cape.png'  # Path to the mask image
    _config.mask.blur = 2.0 # SD for Gaussian blur applied to mask
    _config.bc.function = 'const-outlet-rtd'
    _config.bc.width = 0.08 # Width of the sponge region
    _config.time.dt = 1e-4 # param.dt
    _config.time.save_rate = 500 # param.sim_steps
    _config.time.T = 12
    _config.ic.n_batch = 1
    _config.fps = 4
    autorun(_config)


def test_riot():
    _config = config()
    _config.logging.task_name = "test_riot"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 512
    _config.grid.Ny = 512
    _config.pde.nu = 5e-4 # RE 200
    _config.pde.penalty = 4.0 # Brinkman penalty parameter for cylinder
    _config.pde.B = 20.0 # beta plane
    _config.ic.wavenumbers = [1, 3] # param.data_wavenumbers
    _config.ic.energy = 0.0005
    _config.mask.function = 'netCDF'
    _config.mask.mask = 'riot_070725'  # Path to the mask image
    _config.mask.blur = 2.0 # SD for Gaussian blur applied to mask
    _config.mask.pad = 0.24 # Padding around the mask; should be the same as sponge * 3
    _config.mask.pad_mode = 'trd' # top, right, down only
    _config.bc.function = 'const-outlet-rtd'
    _config.bc.width = 0.08 # Width of the sponge region
    _config.time.dt = 1e-4 # param.dt
    _config.time.save_rate = 500 # param.sim_steps
    _config.time.T = 12
    _config.ic.n_batch = 1
    _config.fps = 4
    autorun(_config)
    
def test_periodic_step():
    _config = config()
    _config.logging.task_name = "test_periodic_step"
    _config.logging.run_name = ""
    _config.forcing = None
    _config.grid.Nx = 256
    _config.grid.Ny = 256
    _config.pde.nu = 5e-4 # RE 200
    _config.pde.penalty = 4.0 # Brinkman penalty parameter for cylinder
    _config.ic.wavenumbers = [1, 3] # param.data_wavenumbers
    _config.ic.energy = 0.0005
    _config.mask.function = 'image'
    _config.mask.mask = 'periodic_step.png'  # Path to the mask image
    _config.mask.blur = 2.0 # SD for Gaussian blur applied to mask
    _config.bc.function = 'const-outlet-rtd'
    _config.bc.width = 0.08 # Width of the sponge region
    _config.time.dt = 1e-4 # param.dt
    _config.time.save_rate = 500 # param.sim_steps
    _config.time.T = 12
    _config.ic.n_batch = 1
    _config.fps = 4
    autorun(_config)