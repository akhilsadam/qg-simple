
from qg.solver.util import _Cache, cached_dot

class AB(_Cache):
    def __init__(self, level=2):
        super().__init__(level,
            cached_dot({
                1: [1],
                2: [3/2, -1/2],
                3: [23/12, -4/3, 5/12],
                4: [55/24, -59/24, 37/24, -3/8],
                5: [1901/720, -1387/360, 109/30, -637/360, 251/720]
            })
        )
 
 
# Application of the Bezier integration technique with enhanced stability 
# in forward dynamics of constrained multibody systems 
# with Baumgarte stabilization method
# Mohammad Khoshnazar et al., 2023 Engineering with Computers
# https://link.springer.com/article/10.1007/s00366-023-01884-x
       
class BZ(_Cache):
    def __init__(self, level=2):
        super().__init__(level,
            cached_dot({
                1: [1],
                2: [3/2, -1/2],
                3: [19/12, -8/12, 1/12],
                4: [175/108, -81/108, 15/108, -1/108],
            })
        )