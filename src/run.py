if __name__ == "__main__":
    
    import sys
    from mura.deploy.run import run
    from mura.deploy.util import cprint
    from qg.solver.qg import QG
    
    param, wandb_logger, py_logger, version, save_path = run(sys.argv)
    py_logger.info(version)
    py_logger.info(save_path)
    py_logger.info(cprint(param))

    QG(param,logger=py_logger).solve(save_path=save_path)