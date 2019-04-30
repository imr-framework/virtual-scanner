# Simulates effect of rf pulses using the bloch equation
# Might be useful for pulse sequence design
#       and testing out RF pulses in pulseq files ...
# consider adding in SLR algorithm later

from math import pi
import bloch as blc
import numpy as np
import matplotlib.pyplot as plt
from pulseq.core.Sequence.sequence import Sequence
from pulseq.core.Sequence.read_seq import read
from pulseq.core.make_sinc import make_sinc_pulse
from pulseq.core.opts import Opts

GAMMA_BAR = 42.58e6

def sim_rf_action(b1,dt,g,locs):
    s = []
    for loc in locs:
        m = np.array([[0], [0], [1]])
        ss = []
        for v in range(len(b1)):
            B1x = np.real(b1[v])
            B1y = np.imag(b1[v])
            A = np.array([[0,g*loc,B1y],
                          [-g*loc,0,B1x],
                          [B1y,-B1x,0]])
            m = m + dt*(GAMMA_BAR*2*pi)*A@m
            ss.append(np.sqrt(m[0]**2 + m[1]**2))
        print("another loc done!")
        s.append(ss)
        plt.figure(100)
        plt.plot(np.arange(0,len(b1)*dt,dt),ss)

    plt.show()

if __name__ == '__main__':
    kwargs_for_opts = {"rf_ring_down_time": 30e-6, "rf_dead_time": 100e-6,"rf_raster_time":1e-7}
    system = Opts(kwargs_for_opts)
    kwargs_for_sinc = {"flip_angle": 90*pi/180, "system": system, "duration": 4e-3, "slice_thickness": 0.01,
                       "apodization": 0.5, "time_bw_product": 4}
    rf, gz = make_sinc_pulse(kwargs_for_sinc, nargout=2)
    b1 = rf.signal[0]/GAMMA_BAR
    dt = system.rf_raster_time
    #plt.figure(1)
    #plt.plot(rf.t[0],np.absolute(b1))
    #plt.show()

    locs = [-0.01,-0.005,0,0.005,0.01]
    S = sim_rf_action(b1,dt,gz.amplitude/GAMMA_BAR,locs)





