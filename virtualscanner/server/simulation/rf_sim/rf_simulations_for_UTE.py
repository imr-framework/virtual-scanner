import virtualscanner.server.simulation.bloch.spingroup_ps as sg
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from virtualscanner.server.simulation.rf_sim.animate_spins import animate_spins
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.opts import Opts
from virtualscanner.server.simulation.rf_sim.rf_helpers import *
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from scipy.io import savemat

from virtualscanner.server.simulation.rf_sim.rf_simulations import simulate_rf

system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

# Sequence components
FA = 10  # deg
thk = 5e-3

# rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA*np.pi/180, duration=1e-3, slice_thickness=thk, apodization=0.5,
#                                  time_bw_product=2, center_pos=1, system=system, return_gz=True)

#rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA * np.pi / 180, duration=2e-3, slice_thickness=thk, apodization=0.5,
 #                                 time_bw_product=2, center_pos=0.5, system=system, return_gz=True)
FA = 30
rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA * np.pi / 180, duration=2e-3, slice_thickness=thk, apodization=0.5,
                                  time_bw_product=2, center_pos=1, system=system, return_gz=True)

GAMMA = 42.58e6 * 2 * pi
GAMMA_BAR = GAMMA / (2 * np.pi)
rf_dt = rf.t[1] - rf.t[0]
print(f'Slice bw : {thk * gz.amplitude} Hz')
bwbw = 2 * thk * gz.amplitude
signals, m = simulate_rf(bw_spins=bwbw, n_spins=200, pdt1t2=(1, 0, 0), flip_angle=90, dt=rf_dt,
                         solver="RK45",
                         pulse_type='custom', pulse_shape=rf.signal / GAMMA_BAR, display=False)



print(m.shape)

savemat('sim_results/2d_rw_UTE_halfpulse_results_fa30.mat', {'m': m[:, :, -1], 'thk_sim': 2 * thk})
