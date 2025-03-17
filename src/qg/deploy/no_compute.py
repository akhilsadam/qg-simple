import qg._input.validate_configuration as vc   
from mura import SingleInstancer
# from mura.deploy.util import serialize_class
from param import param

if __name__ == "__main__":
    
    auto = SingleInstancer(vc.validate, vc.config, param, _action_name='qg', no_compute=True)
    auto.run()