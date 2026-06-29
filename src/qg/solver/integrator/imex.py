
def CN2(u, explicit_step, dt, linear_operator):
    # u_n+1 - dt/2 f(u_n+1) = u_n + dt/2 f(u_n)
    rhs = (0.5 * dt * linear_operator) * u + explicit_step
    lhs = (1 - 0.5 * dt * linear_operator)
    return rhs / lhs
