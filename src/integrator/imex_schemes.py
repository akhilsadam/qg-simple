import torch
import numpy as np

def backward_euler(non_linear_term,dt):
    return non_linear_term*dt

def CN2(linear_operator,input_field,dt):
    # Linear_operator is the object
    return 0.5*dt*linear_operator.apply(input_field),0.5*dt*linear_operator.Lc

def AB2(non_linear_term1,non_linear_term2,dt):
    return (3/2)*(dt)*non_linear_term1  - (1/2)*(dt)*non_linear_term2