import torch
import torch.nn.functional as F
from qg.solver.opt.basis import to_physical, to_spectral, dealias

def sponge(state, derivative, eta, masks, uh, vh):
    
    _outlet_v1, _outlet_v2, _outlet_v1_ramp = masks
    
    _eta = eta * state.dt
    
    ### vorticity / open bc
    outlet_vorticity_sponge = to_spectral(_outlet_v1 * to_physical(1j * derivative.kr * state.vh - 1j * derivative.ky * state.uh)) / _eta
    
    masked_vh_delta = to_spectral(_outlet_v2 * to_physical(state.vh - vh)) 
    masked_uh_delta = to_spectral(_outlet_v2 * to_physical(state.uh - uh))
    
    outlet_velocity_sponge = (1j * derivative.kr * masked_vh_delta - 1j * derivative.ky * masked_uh_delta) / _eta
    
    outlet_diffusion = -0.1 * derivative.krsq * to_spectral(_outlet_v1_ramp * to_physical(state.qh))
    
    
    return outlet_diffusion + outlet_velocity_sponge


def const_x_flow(state, inlet_velocity):
    flow_uh = torch.zeros_like(state.uh)
    flow_uh[...,0,0] = -inlet_velocity # direction flip
    state.uh[...,0,0] = -inlet_velocity # direction flip
    state.vh[...,0,0] = 0.0
    return flow_uh


def vertical_outlet_mask(grid, X, Y, _min, _max, x, width):
    _mask_1 = ((X > x) * (X < x + width) \
        * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
    _mask_2 = ((X > x + width) * (X < x + 2 * width) \
        * (Y >= _min) * (Y <= _max))[None,:,:].to(grid.ftype)
    
    _mask_ramp = _mask_1 \
        * (X - x) / width
        
    return _mask_1, _mask_2, _mask_ramp


def horizontal_bidirectional_mask(grid, X, Y, _min, _max, y, width):
    _mask_1 = ((Y > y) * (Y < y + width) \
        * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
    _mask_2 = ((Y > y + width) * (Y < y + 2 * width) \
        * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
    _mask_3 = ((Y > y + 2 * width) * (Y < y + 3 * width) \
        * (X >= _min) * (X <= _max))[None,:,:].to(grid.ftype)
    
    
    _mask_ramp = \
        _mask_1  * (Y - y) / width + \
        _mask_3 * ((y + 3 * width) - Y) / width
        
    return _mask_1 + _mask_3, _mask_2, _mask_ramp

def const_outlet_r(state, grid, derivative, 
                    inlet_velocity=1.0, _min=0.0, _max=1.0, eta=4.0,
                    width = 0.05,                 
                    **kwargs):

    # right side outlet
    
    x = torch.linspace(0, 1, grid.Nx,device=grid.device)
    y = torch.linspace(0, 1, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None]
        
    flow_uh = const_x_flow(state, inlet_velocity)
    flow_vh = flow_uh * 0.0
    
    ### regions
    _outlet_1, _outlet_2, _outlet_1_ramp = vertical_outlet_mask(grid, X,  Y, _min, _max, 1 - 2 * width, width)
        
    masks = (_outlet_1, _outlet_2, _outlet_1_ramp)
    _sponge = sponge(state, derivative, eta, masks, flow_uh, flow_vh)
    
    return dealias(_sponge, derivative, 1/3)

def const_outlet_rtd(state, grid, derivative, 
                    inlet_velocity=1.0, _min=0.0, _max=1.0, eta=4.0,
                    width = 0.05,                 
                    **kwargs):
    
    # right, top and down outlets
    
    x = torch.linspace(0, 1, grid.Nx,device=grid.device)
    y = torch.linspace(0, 1, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None]
        
    flow_uh = const_x_flow(state, inlet_velocity)
    flow_vh = flow_uh * 0.0
    
    ### regions
    _outlet_1, _outlet_2, _outlet_ramp = vertical_outlet_mask(grid, X,  Y, _min, _max, 1 - 2 * width, width)
    _bidirect_1, _bidirect_2, _bidirect_ramp = horizontal_bidirectional_mask(grid, X, Y, _min, _max, 1 - 3*width, width) # just above lower image boundary
        
    masks = (_outlet_1 + _bidirect_1, _outlet_2 + _bidirect_2, _outlet_ramp + _bidirect_ramp)
    _sponge = sponge(state, derivative, eta, masks, flow_uh, flow_vh)
    
    return dealias(_sponge, derivative, 1/3)


def none(state, grid, derivative,                  
                    **kwargs):
    return 0

####################################################################################################

valid_bc = lambda _bc: hasattr(_bc, 'function') and _bc.function in bc_library

bc_library = {
    'periodic': none,
    'const-outlet-r': const_outlet_r,
    'const-outlet-rtd': const_outlet_rtd,
}

solve_bc = lambda _bc: (lambda *args: bc_library[_bc.function](*args, **_bc.__dict__)) if valid_bc(_bc) else _bc