from qg.config import QGConfig
from qg.solver.qg import QG

from mura import data_run

def autorun(_config, **kwargs):
    """Legacy autorun function."""
    with data_run(_config) as save_path:
        qg = QG(_config)
        qg.solve(save_path=save_path, **kwargs)