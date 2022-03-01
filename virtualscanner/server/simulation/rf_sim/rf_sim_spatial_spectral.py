# Simulate the spatial-spectral response from the MSI sequence in development
from pypulseq.Sequence.sequence import Sequence
import virtualscanner.server.simulation.bloch.spingroup_ps as sg
import numpy as np
from scipy.io import savemat
# Make a grid of spins in spatial-spectral space and apply pulse+gradients / delays
# For now, just do {90(+Gz), delay, 180(-Gz)} (turbo_factor = 1)
GAMMA_BAR = 42.58e6
# Load mock sequence and extract the necessary information
seq = Sequence()

#seq.read('2dmsi_sim.seq')
seq.read('2dmsi_sim_single_slice_19bins_800Hz_per_bin.seq')

# Extract RF(90), delay, RF(180)

# Simulate with NumSolverSpinGroup class

spins_list = []
zs = np.linspace(-15e-3,15e-3,15, endpoint=True) # [meters]
#dfs = np.linspace(-1200,1200,16, endpoint=True) # [Hz]
dfs = np.linspace(-7600, 5400, 16, endpoint=True)

for z in zs:
    for df in dfs:
        newspin = sg.NumSolverSpinGroup(loc=(0, 0, z), pdt1t2=(1,0,0), df=df)
        spins_list.append(newspin)

blk = 4
count = 0
while (blk+3) <= 7: # len(seq.dict_block_events): # Over all slices and all
    #TODO factor in RF phase and frequency displacements.

    rf90 = seq.get_block(blk).rf
    grad90 = seq.get_block(blk).gz
    rf180 = seq.get_block(blk+2).rf
    grad180 = seq.get_block(blk+2).gz
    delay_time = seq.get_block(blk+1).delay.delay

    dt90 = rf90.t[1] - rf90.t[0]
    pulse_shape_90 = rf90.signal/GAMMA_BAR
    pulse_shape_90 *= np.exp(-1j*rf90.phase_offset)*np.exp(-1j*2*np.pi*rf90.freq_offset*rf90.t)
    grads_shape_90 = np.zeros((3, len(pulse_shape_90)))
    grads_shape_90[2,:] = np.interp(dt90*np.arange(len(rf90.signal)),
                                np.cumsum([0,grad90.rise_time,grad90.flat_time,grad90.fall_time]),
                                [0,grad90.amplitude/GAMMA_BAR,grad90.amplitude/GAMMA_BAR,0])

    dt180 = rf180.t[1] - rf180.t[0]
    pulse_shape_180 = rf180.signal/GAMMA_BAR
    pulse_shape_180 *= np.exp(-1j*rf180.phase_offset)*np.exp(-1j*2*np.pi*rf180.freq_offset*rf180.t)
    grads_shape_180 = np.zeros((3, len(pulse_shape_180)))
    grads_shape_180[2,:] = np.interp(dt180*np.arange(len(rf180.signal)),
                                np.cumsum([0,grad180.rise_time,grad180.flat_time,grad180.fall_time]),
                                [0,grad180.amplitude/GAMMA_BAR,grad180.amplitude/GAMMA_BAR,0])
    blk += 3

    spin = spins_list[0]
    results90 = [spin.apply_rf_store(pulse_shape_90, grads_shape_90, dt90)[0][-1] for spin in spins_list]
    __ = [spin.delay(delay_time) for spin in spins_list]
    results180 = [spin.apply_rf_store(pulse_shape_180, grads_shape_180, dt180)[0][-1] for spin in spins_list]


    for spin in spins_list: spin.reset()

    count += 1
    print(f'Pulse #{count} simulated.')

savemat('msi_sim_results_1st_pulse_pair_exp1wideBW_2ndbin_bindisptest.mat',
        {'after_1st': results90, 'after_2nd': results180,'zs':zs, 'dfs':dfs, 'shape':np.array([len(dfs),len(zs)])})





