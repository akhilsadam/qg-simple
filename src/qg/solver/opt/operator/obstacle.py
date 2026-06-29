import torch
import torch.nn.functional as F
import inspect
from qg.solver.opt.basis import to_physical, to_spectral

    
def solve_mask(mask, grid, derivative):
    signature = inspect.signature(mask)
    num_params = len(signature.parameters)
    print(f"Mask function {mask.__name__} has {num_params} parameters.")
    
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


def compute_normal_vectors(op, mask):
    # Compute gradients
    # kernel = 1/16 * torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).to(mask.device)[None,None,...]
    # padded = F.pad(mask[None, :, :, :], (1, 1, 1, 1), mode='circular')
    # mask_b = F.conv2d(padded, kernel, padding=0)[0]
    
    mask_h = op.derivative.blur * to_spectral(mask) 
    
    grad_x = - to_physical(op.derivative.dx * mask_h) * mask * (1-mask)
    grad_y = - to_physical(op.derivative.dy * mask_h) * mask * (1-mask)    
    
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(10,10))
    # plt.imshow(torch.stack([grad_x,grad_y,mask], dim=-1).squeeze().detach().cpu().numpy())
    # plt.savefig("normal_test.png")
    # exit()
    
    # padded_x = F.pad(grad_x[None, :, :, :], (1, 1, 1, 1), mode='circular')
    # grad_x = F.conv2d(padded_x, kernel, padding=0)[0]
    # padded_y = F.pad(grad_y[None, :, :, :], (1, 1, 1, 1), mode='circular')
    # grad_y = F.conv2d(padded_y, kernel, padding=0)[0]
       
    eps = 1e-15

    grad_magnitude = (grad_x**2 + grad_y**2) ** 0.5
    
    # Normalize the gradient to get the unit normal vector
    window = 4 * mask * (1-mask)
    normal_x = grad_x * window / (grad_magnitude + eps)  # Add small epsilon to avoid division by zero
    normal_y = grad_y * window / (grad_magnitude + eps)
    
    return normal_x, normal_y, window, 1-window

def curvature(op, nx, ny):
    return to_physical(op.derivative.dx * to_spectral(nx) + op.derivative.dy * to_spectral(ny))

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
    
    sponge = (- op.derivative.dx * v_chi_h + op.derivative.dy * u_chi_h) / eta # - d/dx(chi*v) + d/dy(chi*u)
    
    return sponge

# def brinkman_friction_slip_penalty(op, state, chi, chi_velocity):
#     eta = op.params.penalty * op.dt

#     u = to_physical(state.uh)
#     v = to_physical(state.vh)
#     q = to_physical(state.qh)

#     nx, ny, interior = compute_normal_vectors(op, chi)
#     kappa = curvature(op, nx, ny) * (1-chi)
    
#     # tangential velocity
#     ut = v * nx - u * ny

#     target = kappa * ut

#     q_pen = chi * (q - target)

#     # from matplotlib import pyplot as plt
#     # plt.figure(figsize=(10,10))
#     # plt.imshow(torch.stack([0*chi,0*chi, chi * kappa], dim=-1).squeeze().detach().cpu().numpy())
#     # plt.savefig("normal_test.png")
#     # exit()

#     return -1 * to_spectral(q_pen) / eta

def brinkman_friction_slip_penalty(op, state, chi, chi_velocity):
    """
    Computes the brinkman volume penalization for mask (chi) in spectral space (h).
    """
    eta = op.params.penalty * op.dt
    friction = op.params.friction

    u = to_physical(state.uh) # convert to physical space
    v = to_physical(state.vh)
    q = to_physical(state.qh)
    
    du = (u - chi_velocity[0])
    dv = (v - chi_velocity[1])
    
    normal_x, normal_y, boundary, interior = compute_normal_vectors(op, chi)
    # k = curvature(op, normal_x, normal_y) * (1-chi)
    
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(10,10))
    # plt.imshow(torch.stack([normal_x,normal_y,interior], dim=-1).squeeze().detach().cpu().numpy())
    # plt.savefig("normal_test.png")
    # exit()

    ndot = du * normal_x + dv * normal_y
    tdot = dv * normal_x - du * normal_y
    dun = ndot * normal_x
    dvn = ndot * normal_y

    dut = (du - dun) # tangentials
    dvt = (dv - dvn)

    u_chi = boundary * dun + (interior * dun + chi * dut) * friction # products to be damped
    v_chi = boundary * dvn + (interior * dun + chi * dvt) * friction
    # q_chi = interior * (q)

    u_chi_h = to_spectral(u_chi)
    v_chi_h = to_spectral(v_chi)
    # q_chi_h = to_spectral(q_chi)
    
    sponge = (-1 * op.derivative.dx * v_chi_h + op.derivative.dy * u_chi_h ) / eta # - d/dx(chi*v) + d/dy(chi*u) # surface sponge
    # - q_chi_h
    return sponge

    
def brinkman_friction_slip_w_pot_penalty(op, state, chi, chi_velocity):
    """
    Computes the brinkman volume penalization for mask (chi) in spectral space (h).
    """
    eta = op.params.penalty * op.dt
    friction = op.params.friction

    u = to_physical(state.uh) # convert to physical space
    v = to_physical(state.vh)
    q = to_physical(state.qh)
    
    du = chi * (u - chi_velocity[0])
    dv = chi * (v - chi_velocity[1])
    dq = chi * q
    
    u_chi_h = to_spectral(du)
    v_chi_h = to_spectral(dv)
    q_chi_h = to_spectral(dq)
    
    v_sponge = (-1 * op.derivative.dx * v_chi_h + op.derivative.dy * u_chi_h) / eta # - d/dx(chi*v) + d/dy(chi*u)
    q_sponge = -1 * q_chi_h / eta
    sponge = v_sponge * friction + q_sponge * (1-friction)
    
    return sponge

    
    # modify flow field inside obstacle
    
    # # get closest point along normal
    # x = torch.arange(0, chi.shape[-1], device=chi.device)
    # y = torch.arange(0, chi.shape[-2], device=chi.device)
    # xc = (torch.round(normal_x) + x).to(torch.int32)
    # yc = (torch.round(normal_y) + y).to(torch.int32)
    # # print(xc.shape)
    # uc = dutr[...,yc,xc]
    # vc = dvtr[...,yc,xc]
    # print(yc.shape)
    # print(dvtr.shape, vc.shape, chi.shape)
 

# def inlet_outlet(op, state,
#                     inlet_velocity=1.0, mn=0.4, mx=0.6, eta=2.0,
#                     inlet_x = 0.12,                 
#                     **kwargs):
    
#     x = torch.linspace(0, 1, op.grid.Nx,device=op.grid.device)
#     y = torch.linspace(0, 1, op.grid.Ny,device=op.grid.device)
#     X, Y = x[None,:],y[:,None]
    
    
#     ### regions
#     total_outlet_width = inlet_x
#     outlet_width = total_outlet_width / 3
#     outlet_xw = outlet_width / 2
#     outlet_xv = outlet_width * 3 / 2
#     outlet_xu = outlet_width * 5 / 2
    
    
#     _outlet_w = ((torch.abs(X - outlet_xw) < outlet_width) \
#         * (Y >= mn) * (Y <= mx))[None,:,:].to(op.grid.ftype)
#     _outlet_v = ((torch.abs(X - outlet_xv) < outlet_width) \
#         * (Y >= mn) * (Y <= mx))[None,:,:].to(op.grid.ftype)
#     _outlet_ramp_v = torch.nn.functional.relu((outlet_width - torch.abs(X - outlet_xv))/outlet_width)
#     _outlet_u = ((torch.abs(X - outlet_xu) < outlet_width) \
#         * (Y >= mn) * (Y <= mx))[None,:,:].to(op.grid.ftype)
#     _outlet_ramp_u = torch.nn.functional.relu((outlet_width - torch.abs(X - outlet_xu))/outlet_width)

#     _eta = 2 * eta * state.dt
    
    
            
#     ### inlet
#     _inlet = _outlet_u
#     _inlet_ramp = _outlet_ramp_u
#     vx = - inlet_velocity    
    
#     u = to_physical(state.uh)
#     u += (vx-u)  * _inlet
#     state.uh = to_spectral(u)
#     state.update_qp()
    
#     ### vorticity / open bc
#     outlet_vorticity_sponge = to_spectral(_outlet_w * to_physical(1j * op.derivative.kr * state.vh - 1j * op.derivative.ky * state.uh)) / _eta
    
#     ### y-velocity / straighten
#     sv = to_physical(state.vh)
#     dv = _outlet_v * _outlet_ramp_v * (sv)
#     v_chi_h = to_spectral(dv)   
#     outlet_y_velocity_smooth_sponge = (1j * op.derivative.kr * v_chi_h) / _eta

#     # su = to_physical(state.uh)
#     # du = _inlet * (su - vx)
#     # u_chi_h = to_spectral(du)
#     # outlet_x_velocity_sponge = (- 1j * derivative.ky * u_chi_h) / _eta
    

#     outlet_sponge = outlet_vorticity_sponge \
#         + outlet_y_velocity_smooth_sponge \
#         # + outlet_x_velocity_sponge   
#     return dealias(outlet_sponge, op.derivative, 1/3)