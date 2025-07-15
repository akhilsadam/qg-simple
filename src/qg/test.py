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

