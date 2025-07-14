
def vortex_stretching(op, state):

    l = op.params.rossby_radius
    vs = - state.ph / l**2
    
    return vs