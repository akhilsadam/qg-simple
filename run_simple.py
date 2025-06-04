import logging, os, sys, importlib, importlib.util

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
py_logger = logging.getLogger('local')
py_logger.setLevel(logging.INFO)
         
module_name = 'param'
module_path = os.path.join(os.getcwd(), 'param.py')

if os.path.exists(module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Now you can use the module
else:
    py_logger.warn(f"File not found: {module_path}")
_config = module


param = _config
config = param.config

save_path = '.'

from mura.deploy.util import cprint
from qg.solver.qg import QG
    

# py_logger.info(version)
py_logger.info(save_path)
py_logger.info(cprint(config))

QG(config,logger=py_logger).solve(save_path=save_path)