import torch
import numpy as np

@staticmethod
def puv(qh, derivative):
    ph = derivative.inv_laplacian * qh
    uh = -1 * derivative.dy * ph
    vh = derivative.dx * ph
    return ph, uh, vh

class _state:
    def __init__(self, qh, dt, flow, derivative):
        self.t = 0.0
        self.dt = dt
        self.derivative = derivative
        self.flow = flow
        
        # potential flow velocities; set by bc
        # self.uh_potential = 0
        # self.vh_potential = 0
        
        self._qh = None
        self._ph = None
        self._uh = None
        self._vh = None
        self._needs_sync = False

        self.qh = qh

    def sync(self):
        self._ph, self._uh, self._vh = puv(self._qh, self.derivative)
        self.flow(self) # apply puv conditions
        
        # self._uh = uh + self.uh_potential
        # self._vh = vh + self.vh_potential
        
        self._needs_sync = False

    def ensure_sync(self):
        if self._needs_sync:
            self.sync()

    @property
    def qh(self):
        return self._qh

    @qh.setter
    def qh(self, qh):
        self._qh = qh
        self._needs_sync = True

    @property
    def ph(self):
        self.ensure_sync()
        return self._ph

    @property
    def uh(self):
        self.ensure_sync()
        return self._uh

    @property
    def vh(self):
        self.ensure_sync()
        return self._vh

    def update_uv(self):
        self.ensure_sync()
        return self._ph, self._uh, self._vh

    def update_t(self):
        self.t += self.dt

    def update_uvt(self):
        self.ensure_sync()
        self.update_t()

    def _out(self):
        return to_physical(self._qh)

    def out(self, cdim=1):
        return torch.stack(
            [to_physical(self._qh),
            to_physical(self.ph),
            to_physical(self.uh),
            # to_physical(self.uh_p),
            to_physical(self.vh)],
            dim=cdim, # assume batched
        )

    # def update_potential_flow(self):    
    #     self.uh = self.uh + self.uh_p + self.x_adv * self.dt
    #     self.vh = self.vh + self.vh_p + self.y_adv * self.dt
        
    # def update_qp(self):
    #     self.qh = - 1j * self.derivative.kr * self.vh + 1j * self.derivative.ky * self.uh
    #     self.ph = self.qh * self.derivative.inv_laplacian
        
    #     uh_w = 1j * self.derivative.ky * self.ph
    #     vh_w = -1j * self.derivative.kr * self.ph
        
    #     self.uh_p = self.uh - uh_w
    #     self.vh_p = self.vh - vh_w
    
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