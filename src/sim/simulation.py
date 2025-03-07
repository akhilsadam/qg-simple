import torch
import numpy as np
import math
from tqdm import tqdm
from Operators.spectral_conversion import to_physical, to_spectral, dealias
from Time_marching.imex_schemes import backward_euler, CN2, AB2


class Simulation:
    def __init__(self, grid, params, spectral_derivative, linear_operator, nonlinear_operator, initial_condition, time_params, mask=None, forcing=None):
        """
        Initialize the simulation parameters.
        """
        self.grid = grid
        self.device = grid.device
        self.params = params
        self.spectral_derivative = spectral_derivative
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.initial_condition = initial_condition
        self.xi = mask # Xi is the mask variable
        self.forcing = forcing
        self.dt = time_params.dt
        self.T = time_params.T
        self.save_interval=time_params.save_int
        self.Nx, self.Ny = grid.Nx, grid.Ny  # Grid resolution
        self.steps = int(self.T / self.dt)  # Number of time steps
        self.q_sol = torch.zeros([self.Nx, self.Ny, int(self.steps/self.save_interval)+1]) # Initialize for only the save times
        self.q_sol[:, :, 0] = to_physical(initial_condition)
        
    def time_step(self):
        """Perform the time-stepping loop."""
        dt = self.dt
        xi = self.xi

        for it_count in range(self.steps - 1):
            
            # Handle initial conditions
            if it_count == 0:
                q_sol_h_1 = to_spectral(self.q_sol[:, :, it_count].squeeze()).cuda()
            else:
                q_sol_h_1 = ans

            # Initialize source term
            source = q_sol_h_1

            # Compute nonlinear terms
            if it_count == 0:
                source_jacobian = backward_euler(self.nonlinear_operator.jacobian_pq(q_sol_h_1), dt)
                source_brinkman = backward_euler(self.nonlinear_operator.brinkman_penalty(xi,q_sol_h_1), dt)
            else:
                #q_sol_h_2 = to_spectral(self.q_sol[:, :, it_count - 1].squeeze()).cuda()
                source_jacobian = AB2(self.nonlinear_operator.jacobian_pq(q_sol_h_1), 
                                      self.nonlinear_operator.jacobian_pq(q_sol_h_2), dt)
                source_brinkman = AB2(self.nonlinear_operator.brinkman_penalty(xi,q_sol_h_1),
                                      self.nonlinear_operator.brinkman_penalty(xi,q_sol_h_2), dt)

            # Compute linear terms
            source_lin, op_lin = CN2(self.linear_operator,q_sol_h_1,dt)

            # Update the source term
            source = source + source_lin + source_jacobian + source_brinkman

            # Apply the linear operator inversion
            operator = (1 - op_lin).cuda()
            ans = source / operator
            
            # Store input as term 2 for AB and CN
            q_sol_h_2=q_sol_h_1

            # Convert back to physical space and store the result for every save interval
            if (it_count+1) % self.save_interval == 0:
                save_index = (it_count + 1) // self.save_interval
                self.q_sol[:, :, save_index] = to_physical(ans)

    def run(self):
        """Run the full simulation."""
        self.time_step()
        return self.q_sol  # Return the solution array
