import torch
import numpy as np
import math

class _state:
    def __init__(self, qh, derivative):
        self.qh = qh
        self.t = 0.0
        self.derivative = derivative
        self.update_uv()
        
    def update_uv(self):
        self.ph = - self.qh * self.derivative.irsq
        self.uh = 1j * self.derivative.ky * self.ph
        self.vh = -1j * self.derivative.kr * self.ph
        
    def out(self, cdim=1):
        return torch.stack(
            [to_physical(self.qh),
            to_physical(self.ph),
            to_physical(self.uh),
            to_physical(self.vh)],
            dim=cdim, # assume batched
        )

def to_physical(spectral_field):
    """
    Convert a spectral field to physical space (inverse FFT).
    """
    return torch.fft.irfftn(spectral_field,dim=(-2, -1),norm='forward')

def to_spectral(physical_field):
    """
    Convert a physical field to spectral space (FFT).
    """
    return torch.fft.rfftn(physical_field,dim=(-2, -1),norm='forward')


def dealias(y, spectral_derivative, dealias_factor=1/3):
    """
    Apply dealiasing to the field based on the ratio (usually 1/3 rule).
    The field's high-frequency components are truncated.
    
    Args:
    - y: tensor in spectral space to apply dealiasing.
    - spectral_derivative: SpectralOperator instance to access ky, kr, and krsq.
    - dealias_factor: factor to apply the dealiasing (default is 1/3).
    
    Returns:
    - y: tensor with high frequencies removed.
    """
    kcut = math.sqrt(2) * (1 - dealias_factor) * min(spectral_derivative.ky.max(), spectral_derivative.kr.max())
    
    # Apply dealiasing: set high-frequency components to zero
    y[torch.sqrt(spectral_derivative.krsq) > kcut] = 0
    
    return y
