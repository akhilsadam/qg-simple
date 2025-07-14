from qg._input.validate_configuration import config
from qg.solver.qg import QG

from mura import data_run

def autorun(_config):
    with data_run(_config) as save_path:
        qg = QG(_config)
        qg.solve(save_path=save_path)