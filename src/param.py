import math

class param:
    
    project_name = 'qg' # usually set in config
    # runner = 'run.py' # usually set in tasks
    
    class grid:
        Nx = 512
        Ny = 512
        Lx = 2 * math.pi
        Ly = 2 * math.pi
        
    class time:
        dt = 1e-3
        T = 100
        save_rate = 50        # (Frequency of .np saves and plots)
        
    class pde:
        mu = 0.0               # (Linear drag)
        nu = 1.025e-5          # (Viscosity coefficient)
        B = 5.0                # (Beta plane)
        nv = 1                 # (Hyperviscous order)
        penalty = 1.25         # (Brinkman penalty parameter)
        friction = 0.0         # (friction coefficient - between no-slip and free-slip)
        
    class ic:
        energy = 0.01
        wavenumbers = [10.0, 32.0]
        seed= 86   

    class forcing:
        function = 'unscaled_cosine'
        A=0.25
        B=4
        C=0
        D=0.25
        E=4
        F=0
                
    class mask:
        function = 'circular'
        r = math.pi/4
        tol = 1e-3