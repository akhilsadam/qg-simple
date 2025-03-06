import math

class grid_params:
    Nx= 512
    Ny= 512
    Lx= 2*math.pi
    Ly= 2*math.pi
    
class time_params:
    dt= 1e-3
    T = 100000 *dt 
    
class pde_params:
    mu = 0 #(Linear drag)
    nu = 3.125e-5  #(Viscosity coefficient)
    B = 0  #(Vicosity)
    nv = 1 #(Hyperviscous order)
    penalty_coeff=5*time_params.dt  # (Brinkman penalty parameter)
    
class ic_params:
    energy= 0.01
    wavenumbers= [10.0, 32.0]
    seed= 86   
    
class mask_params:
    r = math.pi/4
    tol = 1e-3
            
class params:
    grid = grid_params
    time = time_params
    pde = pde_params
    ic = ic_params
    mask = mask_params
    run_number = 3
