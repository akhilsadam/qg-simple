import sys

def run(args):
    
    from mura.deploy.run import Run
    from mura.deploy.util import cprint
    from qg.solver.qg import QG
    
    with Run(args) as info:
        config, wandb_logger, py_logger, version, save_path = info
        
        py_logger.info(version)
        py_logger.info(save_path)
        py_logger.info(cprint(config))
        
        QG(config,logger=py_logger).solve(save_path=save_path)

if __name__ == "__main__":
    run(sys.argv)