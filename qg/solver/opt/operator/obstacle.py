import torch
import torch.nn.functional as F
import inspect
from qg.solver.opt.basis import to_physical, to_spectral, dealias

    
def solve_mask(mask, grid, derivative):
    signature = inspect.signature(mask)
    num_params = len(signature.parameters)
    
    kernel = 1/16 * torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).to(grid.device)[None,None,...]

    if num_params == 3:
        def _mask(op, state):
            basic_mask, mask_vel = mask(state, op.grid, op.derivative)
            chi = F.conv2d(basic_mask[None,:,:,:], kernel, padding='same')[0] # B y x
            vel = F.conv2d(mask_vel[None,:,:,:], kernel, padding='same')[0] # B 2 y x
            return chi, vel
    else:
        mask = mask(grid, derivative)
        chi = F.conv2d(mask[None,:,:,:], kernel, padding='same')[0] # B y x
        vel = torch.zeros([2,]).to(grid.device)
        def _mask(op, state):
            return chi, vel
        
    return _mask    

#####


def compute_normal_vectors(mask):
    # Compute gradients using central differences
    grad_x = torch.gradient(mask, dim=-1) [0]
    grad_y = torch.gradient(mask, dim=-2) [0]
    
    # Compute the magnitude of the gradient
    grad_magnitude = (grad_x**2 + grad_y**2) ** 0.5
    
    # Normalize the gradient to get the unit normal vector
    normal_x = grad_x / (grad_magnitude)  # Add small epsilon to avoid division by zero
    normal_y = grad_y / (grad_magnitude)
    
    # Set normal vectors to zero inside the solid region (where mask == 1)
    normal_x[~torch.isfinite(normal_x)] = 0
    normal_y[~torch.isfinite(normal_y)] = 0
    
    return normal_x, normal_y

#####
    
    
def brinkman_no_slip_penalty(op, state, chi, chi_velocity):
    """
    Computes the brinkman volume penalization for mask (chi) in spectral space (h).
    """
    eta = op.params.penalty * op.dt
    
    u = to_physical(state.uh) # convert to physical space
    v = to_physical(state.vh)

    u_chi = chi * (u - chi_velocity[0]) # products to be damped
    v_chi = chi * (v - chi_velocity[1])

    u_chi_h = to_spectral(u_chi)
    v_chi_h = to_spectral(v_chi)
    
    sponge = (1j * op.derivative.kr * v_chi_h - 1j * op.derivative.ky * u_chi_h) / eta # - d/dx(chi*v) + d/dy(chi*u)
    
    return dealias(sponge, op.derivative, 1/3)

def brinkman_friction_slip_penalty(op, state, chi, chi_velocity):
    """
    Computes the brinkman volume penalization for mask (chi) in spectral space (h).
    """
    eta = op.params.penalty * op.dt
    friction = op.params.friction

    u = to_physical(state.uh) # convert to physical space
    v = to_physical(state.vh)
    
    du = (u - chi_velocity[0])
    dv = (v - chi_velocity[1])
    
    normal_x, normal_y = compute_normal_vectors(chi)
    ndot = du * normal_x + dv * normal_y
    dun = ndot * normal_x
    dvn = ndot * normal_y
    dut = (du - dun)
    dvt = (dv - dvn)
    dutr = dut * (1 - friction)
    dvtr = dvt * (1 - friction)

    u_chi = chi * (dun + dut * friction) # products to be damped
    v_chi = chi * (dvn + dvt * friction)

    u_chi_h = to_spectral(u_chi)
    v_chi_h = to_spectral(v_chi)
    
    sponge = (1j * op.derivative.kr * v_chi_h - 1j * op.derivative.ky * u_chi_h) / eta # - d/dx(chi*v) + d/dy(chi*u)
    
    
    # modify flow field inside obstacle
    
    # get closest point along normal
    x = torch.arange(0, chi.shape[-1], device=chi.device)
    y = torch.arange(0, chi.shape[-2], device=chi.device)
    xc = (torch.round(normal_x) + x).to(torch.int32)
    yc = (torch.round(normal_y) + y).to(torch.int32)
    uc = dutr[...,yc,xc]
    vc = dvtr[...,yc,xc]
    
    u_corr = u * (1 - chi) + chi * uc
    v_corr = v * (1 - chi) + chi * vc
    
    state.uh = to_spectral(u_corr)
    state.vh = to_spectral(v_corr)
    
    return dealias(sponge, op.derivative, 1/3)