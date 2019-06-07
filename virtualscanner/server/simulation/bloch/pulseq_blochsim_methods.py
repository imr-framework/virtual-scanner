import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import time
import phantom as pht
import multiprocessing as mp
import spingroup_ps as sg
import pulseq_library as psl
from math import pi

GAMMA_BAR = 42.5775e6
GAMMA = 2*pi*GAMMA_BAR

def apply_pulseq_commands(isc,seq_info):
    cmds = seq_info['commands']
    pars = seq_info['params']
    for c in range(len(cmds)):
        cstr = cmds[c]
        cpars = pars[c]
        if cstr == 'd': # delay
            isc.delay(t=cpars[0])
        elif cstr == 'p': # rf pulse
            isc.apply_rf(pulse_shape=cpars[0],grads_shape=cpars[1],dt=cpars[2])
        elif cstr == 'r': # Readout
            isc.readout(dwell=cpars[0],n=cpars[1],delay=cpars[2],grad=cpars[3],timing=cpars[4])
        elif cstr == 'g': # free precessiong with gradients
            isc.fpwg(grad_area=cpars[0],t=cpars[1])

def store_pulseq_commands(seq):
    events = seq.block_events
    dt_grad = seq.system.grad_raster_time
    dt_rf = seq.system.rf_raster_time
    seq_params = []

    commands = ''
    # Go through pulseq block by block and store commands
    for key in events.keys():
        event_row = events[key]
        this_blk = seq.get_block(key)

        # Case 1: Delay
        if event_row[0] != 0:
            commands += 'd'
            seq_params.append([this_blk['delay'].delay[0]])
        # Case 2: rf pulse
        elif event_row[1] != 0:
            commands += 'p'
            rf_time = np.array(this_blk['rf'].t[0]) - dt_rf
            df = this_blk['rf'].freq_offset
            b1 = np.multiply(np.exp(-2*pi*1j*df*rf_time),this_blk['rf'].signal/GAMMA_BAR)
            rf_grad, rf_timing, rf_duration = combine_gradients(blk=this_blk, timing=rf_time)
            seq_params.append([b1,rf_grad,dt_rf])

        # Case 3: ADC sampling
        elif event_row[5] != 0:
            commands += 'r'
            adc = this_blk['adc']
            dt_adc = adc.dwell
            delay = adc.delay
            grad, timing, duration = combine_gradients(blk=this_blk, dt=dt_adc, delay=delay)
            seq_params.append([dt_adc,int(adc.num_samples),delay,grad,timing])

        # Case 4: just gradients
        elif event_row[2] != 0 or event_row[3] != 0 or event_row[4] != 0:
            commands += 'g'
            # Process gradients
            fp_grads_area = combine_gradient_areas(blk=this_blk)
            dur = find_precessing_time(blk=this_blk,dt=dt_grad)
            seq_params.append([fp_grads_area,dur])

    seq_info = {'commands':commands, 'params':seq_params,'grad_raster_time':dt_grad}
    return seq_info



def apply_pulseq_old(isc,seq):
    """
    Applies a pulseq Sequence onto a SpinGroup and returns magnetization signals

    INPUTS
    isc - SpinGroup object from spingroup_ps.py
    seq - pulseq Sequence object

    OUTPUTS
    signal - all ADC samples taken during application of seq
    """
    signal = []
    events = seq.block_events

    dt_grad = seq.system.grad_raster_time
    dt_rf = seq.system.rf_raster_time

    # Go through pulseq block by block and simulate
    for key in events.keys():
        event_row = events[key]
        this_blk = seq.get_block(key)

        # Case 1: Delay
        if event_row[0] != 0:
            delay = this_blk['delay'].delay[0]
            isc.delay(delay)

        # Case 2: rf pulse
        elif event_row[1] != 0:
            # Later: add ring down and dead time to be more accurate?
            rf_time = np.array(this_blk['rf'].t[0]) - dt_rf
            df = this_blk['rf'].freq_offset
            b1 = np.multiply(np.exp(-2*pi*1j*df*rf_time),this_blk['rf'].signal/GAMMA_BAR)
            rf_grad, rf_timing, rf_duration = combine_gradients(blk=this_blk, timing=rf_time)

            isc.apply_rf(b1,rf_grad,dt_rf)

        # Case 3: ADC sampling
        elif event_row[5] != 0:
            adc = this_blk['adc']
            signal_1D = []
            dt_adc = adc.dwell
            delay = adc.delay
            grad, timing, duration = combine_gradients(blk=this_blk, dt=dt_adc, delay=delay)

            isc.fpwg(grad[:,0]*delay,delay)
            v = 1
            for q in range(1,len(timing)):
                if v <= int(adc.num_samples):
                    signal_1D.append(isc.get_m_signal())
                isc.fpwg(grad[:,v]*dt_adc,dt_adc)
                v += 1
            signal.append(signal_1D)

        # Case 4: just gradients
        elif event_row[2] != 0 or event_row[3] != 0 or event_row[4] != 0:
            # Process gradients
            fp_grads_area = combine_gradient_areas(blk=this_blk)
            dur = find_precessing_time(blk=this_blk,dt=dt_grad)
            isc.fpwg(fp_grads_area,dur)
    return signal



def sim_single_spingroup_old(loc_ind,freq_offset,phantom,seq):
    sgloc = phantom.get_location(loc_ind)
    isc = sg.SpinGroup(loc=sgloc, pdt1t2=phantom.get_params(loc_ind), df=freq_offset)
    signal = apply_pulseq_old(isc,seq)
    return signal


def sim_single_spingroup(loc_ind,freq_offset,phantom,seq_info):
    sgloc = phantom.get_location(loc_ind)
    isc = sg.SpinGroup(loc=sgloc,pdt1t2=phantom.get_params(loc_ind),df=freq_offset)
    apply_pulseq_commands(isc,seq_info)
    return isc.signal


# Helpers
def combine_gradient_areas(blk):
    """
    Combines gradient areas in a pulseq Block
    INPUTS
    blk - pulseq Block obtained from seq.get_block()
    OUTPUT:
    Array [Ax, Ay, Az] - x, y, and z gradient areas. Units: [sec*T/m]
    """
    grad_areas = []
    for g_name in ['gx','gy','gz']:
        if blk.__contains__(g_name):
            g = blk[g_name]
            g_area = g.area if g.type == 'trap' else np.trapz(y=g.waveform, x=g.t)
            grad_areas.append(g_area)
        else:
            grad_areas.append(0)
    return np.array(grad_areas)/GAMMA_BAR


def combine_gradients(blk,dt=0,timing=(),delay=0):
    # Key method!
    """
    Interpolate x, y, and z gradients starting from time 0
    at dt intervals, for as long as the longest gradient lasts
    and combine them into a 3 x N array
    INPUTS:
    blk - pulseq Block obtained from seq.get_block()
    dt* - raster time used in interpolating gradients
    timing* - timing points at which gradients are interpolated
        *Note: only one option, dt or timing, is expected as input
    delay - if nonzero, this adds an additional interval = delay at the beginning of the interpolation
            Currently only used in ADC sampling to capture ADC delay
    """
    grad_timing = []
    duration = 0
    if dt != 0:
        duration = find_precessing_time(blk,dt)
        #grad_timing = np.arange(0,duration+dt,dt) # TODO fix
        grad_timing = np.concatenate(([0],np.arange(delay,duration+dt,dt)))
    elif len(timing) != 0:
        duration = timing[-1] - timing[0]
        grad_timing = timing

    grad = []

    # Interpolate gradient values at desired time points
    for g_name in ['gx','gy','gz']:
        if blk.__contains__(g_name):
            g = blk[g_name]
            g_time, g_shape = ([0, g.rise_time, g.rise_time + g.flat_time, g.rise_time + g.flat_time + g.fall_time],
                               [0,g.amplitude/GAMMA_BAR,g.amplitude/GAMMA_BAR,0]) if g.type == 'trap'\
                               else (g.t, g.waveform/GAMMA_BAR)
            g_time = np.array(g_time)
            grad.append(np.interp(x=grad_timing,xp=g_time,fp=g_shape))
        else:
            grad.append(np.zeros(np.shape(grad_timing)))

    return np.array(grad), grad_timing, duration




def find_precessing_time(blk,dt):
    """
    Finds and returns longest duration among Gx, Gy, and Gz
        for use in SpinGroup.fpwg()
    INPUTS
    blk - pulseq Block obtained from seq.get_block()
    dt - gradient raster time for calculating duration of only arbitrary gradients ('grad' instead of 'trap')
    OUTPUT
    Maximum gradient time among the three gradients Gx, Gy, and Gz
    """
    grad_times = []
    for g_name in ['gx','gy','gz']:
        if blk.__contains__(g_name):
            g = blk[g_name]
            tg = (g.rise_time + g.flat_time + g.fall_time) if g.type == 'trap' else len(g.t[0])*dt
            grad_times.append(tg)
    return max(grad_times)


# B0 map!
def get_dB0_map(maptype=0):
    if maptype == 1:
        # Linear field (~gradient)
        def df(loc):
            b0_sc = 1e-4 # TODO what's a good value?
            return b0_sc * np.sqrt(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2])
    elif maptype == 2:
        # Quadratic field
        def df(loc):
            b0_sc = 1e-3  # TODO what's a good value?
            return b0_sc * (loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2])
    else:
        def df(loc):
            return 0
    return df

# Main program
# Parallel simulation