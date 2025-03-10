import _input.validate_configuration as vc   
from mura import SingleInstancer   
from run.param import param

if __name__ == "__main__":
    auto = SingleInstancer(vc.validate, vc.config, param)
    print(auto)
    auto.run()