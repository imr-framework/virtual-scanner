import numpy as np
GAMMA = 2*42.58e6 * np.pi
GAMMA_BAR = 42.58e6


class SpinGroup:
    """
    Basic unit for pulseq-based simulation
    Contains location, proton density, T1, T2, and off-resonance info
    """

    def __init__(self, loc=(0,0,0), pdt1t2=(1,0,0), df=0):
        """
        Define a spin group (with defined T1,T2,PD,location and frequency offset)
        loc: physical location w/ respect to isocenter (m)
        params: (PD,T1,T2) (a.u., sec, sec)
        df: off-resonance (Hz)
        """

        self.m = np.array([[0], [0], [1]])
        self.PD = pdt1t2[0]
        self.T1 = pdt1t2[1]
        self.T2 = pdt1t2[2]
        self.loc = loc
        self.df = df
        self.signal=[]

    def get_m_signal(self):
        """
        Returns complex rep. of Mxy, scaled by PD
        """
        return np.squeeze(self.PD*(self.m[0] + 1j * self.m[1]))

    def fpwg(self,grad_area,t):
        """
        free precession with gradients for the duration of the longest gradient in the list
        INPUT: grads: list of Gradient objects [g1, g2, g3, ...]

        duration of precession  = duration of longest gradient
        amount of phase change = all 3D gradients combined

        """
        x,y,z = self.loc
        phi = GAMMA*(x*grad_area[0]+y*grad_area[1]+z*grad_area[2])+2*np.pi*self.df*t
        C, S = np.cos(phi), np.sin(phi)
        E1 = 1 if self.T1 == 0 else np.exp(-t/self.T1)
        E2 = 1 if self.T2 == 0 else np.exp(-t/self.T2)
        A = np.array([[E2*C, E2*S, 0],
                      [-E2*S, E2*C, 0],
                      [0, 0, E1]])
        self.m = A@self.m + [[0],[0],[1 - E1]]

    def delay(self, t):
        """
        Applies a time passage to the spin group
        i.e. T1, T2 relaxation with no gradients
            INPUTS: t (s) - delay interval
        """
        self.T1 = max(0,self.T1)
        self.T2 = max(0,self.T2)
        E1 = 1 if self.T1 == 0 else np.exp(-t/self.T1)
        E2 = 1 if self.T2 == 0 else np.exp(-t/self.T2)

        A = np.array([[E2, E2, 0],
                      [-E2, E2, 0],
                      [0, 0, E1]])
        self.m = A@self.m + np.array([[0], [0], [1 - E1]])



    def apply_rf(self, pulse_shape, grads_shape, dt):
        """
        Applies a rf pulse to the spin group
        Simulation method: numerical integration of Bloch equation with B1 field and arbitrary gradient field

        INPUTS
        # pulse_shape : 1 x n complex array (B1)[tesla]
        # grads_shape : 3 x n real array  [tesla/meter]
        # dt: raster time for both shapes [seconds]
        """
        m = self.m
        x,y,z = self.loc
        for v in range(len(pulse_shape)):
            B1 = pulse_shape[v]
            B1x = np.real(B1)
            B1y = np.imag(B1)
            glocp = grads_shape[0,v]*x+grads_shape[1,v]*y+grads_shape[2,v]*z
            A = np.array([[0, glocp, B1y],
                          [-glocp, 0, B1x],
                          [B1y, -B1x, 0]])
            m = m + dt*GAMMA*A@m
        self.m = m

    def apply_rf_old(self, pulse_shape, grads_shape, dt):
        """
        Applies a rf pulse to the spin group
        Simulation method: hard-pulse approximation across small time intervals
        With any gradient

        INPUTS
        # pulse_shape : 1 x n complex array (B1)[tesla]
        # grads_shape : 3 x n real array  [tesla/meter]
        # dt: raster time for both shapes [seconds]
        """

        b1 = pulse_shape
        gs = grads_shape
        loc = self.loc
        for k in range(len(b1)):
            bx = np.real(b1[k])
            by = np.imag(b1[k])
            bz = np.sum(np.multiply([gs[0, k], gs[1, k], gs[2, k]], loc)) + self.df / GAMMA_BAR
            be = np.array([bx, by, bz])
            self.m = anyrot(GAMMA * be * dt) @ self.m

    def readout_old(self,dt,n,delay,grad,timing): # Added 4/23/19
        signal_1D = []
        self.fpwg(grad[:,0]*delay, delay)
        v = 1
        for q in range(1, len(timing)):
            if v <= n:
                signal_1D.append(self.get_m_signal())
            self.fpwg(grad[:, v]*dt,dt)
            v += 1

        self.signal.append(signal_1D)

    def readout(self,dwell,n,delay,grad,timing): # TODO
        signal_1D = []
        # ADC delay
        self.fpwg(np.trapz(y=grad[:,0:2], x=timing[0:2]), delay)
        for q in range(1, len(timing)):
            if q <= n:
                signal_1D.append(self.get_m_signal())
            self.fpwg(np.trapz(y=grad[:,q:q+2], dx=dwell), dwell)

        self.signal.append(signal_1D)

# Helpers
def anyrot(v):
    """ Returns Rodrigues's formula: 3 x 3 matrix for arbitrary rotation
        Around the direction of v and for norm(v) rads
        inputs
            v: 3-tuple (vx,vy,vz)
    """
    vx = v[0]
    vy = v[1]
    vz = v[2]
    th = np.linalg.norm(v,2)
    C = np.cos(th)
    S = np.sin(th)

    if th != 0:
        R = (1/(th*th))*np.array([[vx*vx*(1-C)+th*th*C, vx*vy*(1-C)-th*vz*S, vx*vz*(1-C)+th*vy*S],
                                  [vx*vy*(1-C)+th*vz*S, vy*vy*(1-C)+th*th*C, vy*vz*(1-C)-th*vx*S],
                                  [vx*vz*(1-C)-th*vy*S, vy*vz*(1-C)+th*vx*S, vz*vz*(1-C)+th*th*C]])
    else:
        R = np.array([[1,0,0],[0,1,0],[0,0,1]])

    return R
