import logging, os, sys, importlib, importlib.util

# only runs one file

save_path = '../simple_run/'
logfile = os.path.join(save_path, 'python.log')
os.makedirs(save_path, exist_ok=True)

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
py_logger = logging.getLogger('local')
py_logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
py_logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler(logfile)
fileHandler.setFormatter(logFormatter)
py_logger.addHandler(fileHandler)
         
module_name = 'param'
module_path = os.path.join(os.getcwd(), 'param.py')

if os.path.exists(module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Now you can use the module
else:
    py_logger.warning(f"File not found: {module_path}")
    
_config = module
if hasattr(_config, 'config'):
    cconfig = _config.config
    param = cconfig.actions[0].tasks[0].runs[0]
else:
    print(_config)
    param = _config.param
    
config = param

from mura.deploy.util import cprint

py_logger.info(save_path)
py_logger.info(cprint(config))

from qg.solver.qg import QG
QG(config,logger=py_logger).solve(save_path=save_path)