# Try to simulate single pulse used in MSI
# For debugging!
import numpy as np
from pypulseq.Sequence.sequence import Sequence
import virtualscanner.server.simulation.bloch.spingroup_ps as sg
from virtualscanner.server.simulation.rf_sim.rf_simulations import simulate_rf
from scipy.io import savemat
#
GAMMA_BAR = 42.58e6
#
# testspin = sg.NumSolverSpinGroup(loc=(0,0,0), pdt1t2=(1,0,0), df=0)
seq = Sequence()
seq.read('2dmsi_sim_single_slice_19bins_800Hz_per_bin.seq')
#
blk = 1
rf90 = seq.get_block(blk).rf
grad90 = seq.get_block(blk).gz
rf180 = seq.get_block(blk+2).rf
grad180 = seq.get_block(blk+2).gz
delay_time = seq.get_block(blk+1).delay.delay
# #
dt90 = rf90.t[1] - rf90.t[0]
# pulse_shape_90 = rf90.signal/GAMMA_BAR
# grads_shape_90 = np.zeros((3, len(pulse_shape_90)))
# grads_shape_90[2,:] = np.interp(dt90*np.arange(len(rf90.signal)),
#                         np.cumsum([0,grad90.rise_time,grad90.flat_time,grad90.fall_time]),
#                         [0,grad90.amplitude/GAMMA_BAR,grad90.amplitude/GAMMA_BAR,0])
# #
dt180 = rf180.t[1] - rf180.t[0]
# pulse_shape_180 = rf180.signal/GAMMA_BAR
# grads_shape_180 = np.zeros((3, len(pulse_shape_180)))
# grads_shape_180[2,:] = np.interp(dt180*np.arange(len(rf180.signal)),
#                         np.cumsum([0,grad180.rise_time,grad180.flat_time,grad180.fall_time]),
#                         [0,grad180.amplitude/GAMMA_BAR,grad180.amplitude/GAMMA_BAR,0])
#
# results90 = testspin.apply_rf_store(pulse_shape_90, grads_shape_90, dt90)[0]
# testspin.delay(delay_time)
# results180 = testspin.apply_rf_store(pulse_shape_180, grads_shape_180, dt180)[0]
#
# print('Results after 1st pulse: ', results90)
# print('Results after 2nd pulse: ', results180)


# Simulate slice profiles of 90 and 180 pulses used in current MSI (v1) sequence
GAMMA_BAR = 42.58e6
thk = 5e-3
print(f'Slice bw for 90-deg: {thk * grad90.amplitude} Hz')
signals, m90 = simulate_rf(bw_spins=np.absolute(thk * grad90.amplitude), n_spins=100, pdt1t2=(1,0,0), flip_angle=90,
                                      dt=dt90,
                                      solver="RK45",
                                      pulse_type='custom', pulse_shape=rf90.signal / GAMMA_BAR, display=False)


GAMMA_BAR = 42.58e6
thk = 5e-3
print(f'Slice bw for 180-deg: {thk * grad180.amplitude} Hz')
signals, m180 = simulate_rf(bw_spins=np.absolute(thk * grad180.amplitude), n_spins=100, pdt1t2=(1,0,0), flip_angle=180,
                                      dt=dt180,
                                      solver="RK45",
                                      pulse_type='custom', pulse_shape=rf180.signal / GAMMA_BAR, display=False)

savemat('msi_slice_profile_simmed.mat',{'satpulse': m90, 'refpulse': m180})