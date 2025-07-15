import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from qg.solver.opt.basis import _state
from qg.solver.integrator.imex import CN2, AB2

import qg._input.validate_configuration as vc

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.operator import ImplicitLinearOperator, define_explict_operator

from qg.solver.opt.operator.jacobian import advection_uv

import os
import logging
import jpcm.draw as draw

# TODO enable float32/64 precision
# TODO enable Sponge

class QG():
    def __init__(self, param,
                 grid = CartesianGrid,
                 derivative = Derivative,
                 implicit_linear_operator = ImplicitLinearOperator,
                 explicit_sources = [],
                 logger = logging.getLogger(__name__)):
        
        param = vc.validate(param).solve()

        self.param = param
        self.logger = logger
        
        self.grid = grid(**param.grid.__dict__)
        self.derivative = derivative(self.grid)
        self.implicit_linear_operator = implicit_linear_operator(self.grid, self.derivative, param.pde)
        self.operator = define_explict_operator(param, self.grid, self.derivative, self.logger,
                                        args=(param.time.dt, self.grid, self.derivative, param.pde),
                                        sources=explicit_sources) 
        
        self.dt = param.time.dt
        self.logger.info(f"Initialized QG model with {self.grid.Nx}x{self.grid.Ny} grid on {self.grid.device}")

    def step(self, state):
        state.dt = self.dt # Not sure if this is necessary, need to think about adaptive time stepping TODO

        # vorticity step
        explicit_source = AB2(self.operator.source(state)) # source term
        state.qh = CN2(state.qh, explicit_source, self.dt, self.implicit_linear_operator) # Crank-Nicolson        
        
        # potential flow velocity step
        # state.x_adv, state.y_adv = advection_uv(self.operator, state)
        
        # update fields
        state.update_uv()
        # state.update_potential_flow() # also potential_flow
        state.update_t()

    def init(self):  
        return _state(self.param.ic(self.grid, self.derivative), self.dt, self.derivative) # In spectral space
          
    def _run(self):
        save_rate = self.param.time.save_rate
        steps = int(self.param.time.T / self.dt)  # Number of time steps
        solution = torch.zeros([int(steps/save_rate)+1, 4, self.grid.Nx, self.grid.Ny])
        
        state = self.init()
        for it in tqdm(range(steps - 1)):
            self.step(state)            
            
            if (it+1) % save_rate == 0:
                save_index = (it + 1) // save_rate
                solution[save_index, ...] = state.out() # B T C H W
                
        return solution
    
    def solve(self, save_path): # for direct user call
        solution_torch = self._run()
        solution = solution_torch.cpu().numpy()
        self.logger.info(f"Simulation complete.")
        
        np.save(os.path.join(save_path,'DNS.npy'), solution)
        self.logger.info(f"Simulation saved at {save_path}")
        
        # select a couple batches for visualization (permute 0,1 axes)
        solution_b = np.transpose(solution[0:4, ...],(1,0,2,3,4))  # T (selected_B) C H W

        draw.mp4(os.path.join(save_path,'DNS.mp4'), solution_b,
                   fps=20, triplet=True)
        draw.mp4(os.path.join(save_path,'DNS_clamped.mp4'), solution_b,
                   fps=20, triplet=True, clamp=0.3)
        draw.mp4(os.path.join(save_path,'DNS_seismic.mp4'), solution_b,
                   fps=20, triplet=True, cmap='seismic', clamp=0.3)  
        
        # draw.mp4(os.path.join(save_path,'DNS.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1])
        # draw.mp4(os.path.join(save_path,'DNS_clamped.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1], clamp=0.3)
        # draw.mp4(os.path.join(save_path,'DNS_seismic.mp4'), solution_b,
        #            fps=20, triplet=False, mn = [4,1], cmap='seismic', clamp=0.3)       
        self.logger.info(f"Videos saved.")
        
        return solution_torch
