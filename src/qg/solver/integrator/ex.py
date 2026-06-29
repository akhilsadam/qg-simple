# can add RK here

def Euler(u, state, dt, explicit_operator):
    return u + dt * explicit_operator(state)