from qg.solver.opt.basis import to_physical, to_spectral, dealias

def jacobian_pq(op, state):
    """
    Computes the Jacobian of q (vorticity) and p (streampatch) in spectral space (h).
    """
    q = to_physical(state.qh)
    u = to_physical(state.uh)
    v = to_physical(state.vh)

    uq = u * q # calculate products
    vq = v * q

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)
    
    jacobian = 1j * op.derivative.kr * uqh + 1j * op.derivative.ky * vqh # - d/dx(u*q) - d/dy(v*q)
    
    return dealias(
        jacobian,  
        op.derivative,
        1/3)