import torch
import torch.nn.functional as F
from qg.solver.opt.basis import to_physical, to_spectral, dealias

def inlet_outlet(state, grid, derivative, 
                    inlet_velocity=1.0, mn=0.4, mx=0.6, eta=2.0,
                    inlet_x = 0.12,                 
                    **kwargs):
    
    x = torch.linspace(0, 1, grid.Nx,device=grid.device)
    y = torch.linspace(0, 1, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None]
    
    
    ### regions
    total_outlet_width = inlet_x
    outlet_width = total_outlet_width / 3
    outlet_xw = outlet_width / 2
    outlet_xv = outlet_width * 3 / 2
    outlet_xu = outlet_width * 5 / 2
    
    
    _outlet_w = ((torch.abs(X - outlet_xw) < outlet_width) \
        * (Y >= mn) * (Y <= mx))[None,:,:].to(grid.ftype)
    _outlet_v = ((torch.abs(X - outlet_xv) < outlet_width) \
        * (Y >= mn) * (Y <= mx))[None,:,:].to(grid.ftype)
    _outlet_ramp_v = torch.nn.functional.relu((outlet_width - torch.abs(X - outlet_xv))/outlet_width)
    _outlet_u = ((torch.abs(X - outlet_xu) < outlet_width) \
        * (Y >= mn) * (Y <= mx))[None,:,:].to(grid.ftype)
    _outlet_ramp_u = torch.nn.functional.relu((outlet_width - torch.abs(X - outlet_xu))/outlet_width)

    _eta = 2 * eta * state.dt
    
    
            
    ### inlet
    _inlet = _outlet_u
    _inlet_ramp = _outlet_ramp_u
    vx = - inlet_velocity    
    
    u = to_physical(state.uh)
    u += (vx - u)  * _inlet  #- u * _inlet_ramp
    state.uh = to_spectral(u)
    state.update_qp()
    
    ### vorticity / open bc
    outlet_vorticity_sponge = to_spectral(_outlet_w * to_physical(1j * derivative.kr * state.vh - 1j * derivative.ky * state.uh)) / _eta
    
    ### y-velocity / straighten
    sv = to_physical(state.vh)
    dv = _outlet_v * _outlet_ramp_v * (sv)
    v_chi_h = to_spectral(dv)   
    outlet_y_velocity_smooth_sponge = (1j * derivative.kr * v_chi_h) / _eta

    # su = to_physical(state.uh)
    # du = _inlet * (su - vx)
    # u_chi_h = to_spectral(du)
    # outlet_x_velocity_sponge = (- 1j * derivative.ky * u_chi_h) / _eta
    

    outlet_sponge = \
        + outlet_y_velocity_smooth_sponge \
        + outlet_vorticity_sponge # + outlet_x_velocity_sponge   
    return dealias(outlet_sponge, derivative, 1/3)

####################################################################################################

valid_bc = lambda _bc: hasattr(_bc, 'function') and _bc.function in bc_library

bc_library = {
    'inlet-outlet': inlet_outlet,
}

solve_bc = lambda _bc: (lambda *args: bc_library[_bc.function](*args, **_bc.__dict__)) if valid_bc(_bc) else _bc