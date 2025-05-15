if __name__ == "__main__":
    
    import sys
    from mura.deploy.run import Run
    from mura.deploy.util import cprint
    from qg.solver.qg import QG
    
    with Run(sys.argv) as info:
        param, wandb_logger, py_logger, version, save_path = info
        py_logger.info(version)
        py_logger.info(save_path)
        py_logger.info(cprint(param))
        QG(param,logger=py_logger).solve(save_path=save_path)