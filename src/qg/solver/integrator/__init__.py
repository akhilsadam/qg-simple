from .imex import CN2
from .ex import Euler
from .lms import AB, BZ

def select_lms(name):
    if 'AB' in name:
        return AB(int(name.split('AB')[1]))
    elif 'BZ' in name:
        return BZ(int(name.split('BZ')[1]))
    else:
        return AB(2)

def select_ex(name):
    return {
        'Euler': Euler
    }.get(name, Euler)
    
def select_imex(name):
    return {
        'CN2':CN2
    }.get(name, CN2)
    
class Integrator():
    def __init__(self, param):
        self._lms = select_lms(param.lms)
        self._ex = select_ex(param.ex)
        self._imex = select_imex(param.imex)
        # linear multistep, explicit, implicit-explicit
        
    def imex(self, u, state, dt, explicit_source_opt, implicit_linear_opt):
        # where is AB in here? before after in the middle?
        # traditionally before...but seems after can also be useful
        explicit_source = lambda x : self._lms(explicit_source_opt(x))
        explicit_step = self._ex(u, state, dt, explicit_source)
        return self._imex(u, explicit_step, dt, implicit_linear_opt)      
        
    def ex(self, u, state, dt, explicit_source_opt):
        explicit_source = lambda x : self._lms(explicit_source_opt(x))
        return self._ex(u, state, dt, explicit_source)