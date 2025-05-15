import qg._input.validate_configuration as vc   
from mura import Instancer
# from mura.deploy.util import serialize_class
from param import config

if __name__ == "__main__":
    
    auto = Instancer(vc.validate, config=config, gconfig=vc.gconfig)     
    # print(auto)
    # serialize_class(auto, 'test_param.py')
    
    auto.run()