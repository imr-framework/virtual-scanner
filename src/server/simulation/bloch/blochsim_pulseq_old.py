# Simulate a pulseq file directly!
# All for 1 spin group!!!!!!

import bloch as blc
import phantom as pht
from math import pi
import numpy as np
from pulseq.core.Sequence.sequence import Sequence
from pulseq.core.Sequence.read_seq import read
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

# TODO key function!!!!!!
def apply_pulseq_to(isc,seq):
    GAMMA_BAR = 42.58e6
    signal = []
    events = seq.block_events

    # Go through pulseq block by block and simulate!
    for key in events.keys():
        event_row = events[key]
        #print(event_row)
        this_blk = seq.get_block(key)

        # Decide which type of event it is
        # Case 1: Delay
        if event_row[0] != 0:
            delay = this_blk['delay'].delay[0]
            isc.delay(delay)
           # print('Delay by:', delay, 'seconds')

        # Case 2: RF pulse
        elif event_row[1] != 0:
            # TODO later: add ring down and dead time to be more accurate
            # Process RF
            b1 = this_blk['rf'].signal/GAMMA_BAR
            rf_time = this_blk['rf'].t[0]
            rf_pulse = blc.RFPulse(shape=b1,timing=rf_time,raster_time=seq.system.rf_raster_time)
            # Process gradients
            rf_grad = process_gradients(blk=this_blk,grad_raster_time=seq.system.grad_raster_time)
            # Apply rf pulse along with gradients
            isc.apply_rf(rf_pulse,rf_grad)
            print('old')
            print(isc.get_m())
          #  print('Apply RF pulse: ', rf_pulse.get_flip_angle(), 'rad')



        # Case 3: ADC sampling
        elif event_row[5] != 0:
            adc = this_blk['adc']
            ro_grad = process_gradients(blk=this_blk,grad_raster_time=seq.system.grad_raster_time)
            signal.append(isc.readout_gen(ro_grad,blc.ADC(num_samples=int(adc.num_samples),
                                                          dwell_time=adc.dwell,
                                                delay=adc.delay)))
           # print('ADC sampling with grad ', ro_grad.get_shape(),' and n=', adc.num_samples, 'and dt=', adc.dwell,
            #      ' and delay=', adc.delay)


        # Case 4: fpwg
        elif event_row[2] != 0 or event_row[3] != 0 or event_row[4] != 0:
            # Process gradients
            fp_grad = process_gradients(blk=this_blk,grad_raster_time=seq.system.grad_raster_time)
            isc.fpwg([fp_grad])
            #print('Apply gradient with area*gamma_bar=: ', isc.GAMMA_BAR*fp_grad.get_area(), 'm^-1')

    return signal



def sim_single_sg(loc_ind,freq_offset,phantom,seq):
    #print('location:')
    #print(phantom.get_location(loc_ind))
    #print('params:')
    #print(phantom.get_params(loc_ind))
    isc = blc.SpinGroup(loc=phantom.get_location(loc_ind), params=phantom.get_params(loc_ind), df=freq_offset)
    return apply_pulseq_to(isc,seq)


def process_gradients(blk,grad_raster_time):
    """
    Processes gradients (gx,gy,gz) from pulseq block
    and returns a single Gradient object for use in simulator
    INPUTS
        blk: pulseq block object (from .get_block())
        grad_raster_time: raster time for gradients (use the pulseq system's parameter)
    OUTPUT:
        combined Gradient object (could be TrapzGradient or VaryingGradient)
    """

    GAMMA_BAR = 42.58e6

    amp_vec_dict = {'gx':np.array([1,0,0]),
                    'gy':np.array([0,1,0]),
                    'gz':np.array([0,0,1])}
    grad_list = []
    names = ['gx','gy','gz']
    for k in blk.keys():
        for n in names:
            if k == n:
                gi = blk[k]
                if gi.type == 'trap':
                    g = blc.TrapzGradient(rise_time=gi.rise_time, flat_time=gi.flat_time,
                                          fall_time=gi.fall_time, amplitude=gi.amplitude*amp_vec_dict[n]/GAMMA_BAR)
                    grad_list.append(g)
                elif gi.type == 'grad':
                    g = blc.VaryingGradient(shape=gi.waveform*amp_vec_dict[n]/GAMMA_BAR, timing=gi.t)
                    grad_list.append(g)
                else:
                    raise ValueError("Gradient type unidentifiable")

    # Combine gradients
    return combine_gradients_xyz(grad_list,grad_raster_time)


def combine_gradients_xyz(grad_list,raster_time):
    """
    Helper method for process_gradients()
    Combines up to 3 Gradient objects into one
    """
    # Combines x,y,z gradients
    q = len(grad_list)
    # Special case 1: only 1 gradient
    if q == 1:
        return grad_list[0]

    # Special case 2: trapz gradients with identical timing
    elif all(list(map(lambda x: isinstance(x,blc.TrapzGradient),grad_list)))\
        and len(set(list(map(lambda x: x.get_rise_time(),grad_list)))) == 1 \
        and len(set(list(map(lambda x: x.get_flat_time(),grad_list)))) == 1 \
        and len(set(list(map(lambda x: x.get_fall_time(),grad_list)))) == 1:

        return blc.TrapzGradient(rise_time=grad_list[0].get_rise_time(),
                                 flat_time=grad_list[0].get_flat_time(),
                                 fall_time=grad_list[0].get_fall_time(),
                                 amplitude=np.sum(list(map(lambda x:x.get_amplitude(),grad_list)),axis=0))

    # General case: rasterize everything
    else:
        # Take timing to be the longest duration
        dt = raster_time
        grad_timing = np.arange(0,max(list(map(lambda x:x.get_timing()[-1]-x.get_timing()[0],grad_list)))+dt,dt)
        grad_shape = np.zeros((3,len(grad_timing)))

        for u in range(q):
            # Case of trapezoidal gradient
            if isinstance(grad_list[u],blc.TrapzGradient):
                t0 = grad_list[u].get_timing()[0]
                t1 = grad_list[u].get_timing()[-1]
                m = len(np.arange(t0,t1+dt,dt))
                # Add to Gx
                grad_shape[0,0:m] += np.interp(x=np.arange(t0,t1+dt,dt),
                                                xp=grad_list[u].get_timing(),
                                                fp=grad_list[u].get_shape()[:,0])
                # Add to Gy
                grad_shape[1,0:m] += np.interp(x=np.arange(t0,t1+dt,dt),
                                                xp=grad_list[u].get_timing(),
                                                fp=grad_list[u].get_shape()[:,1])
                # Add to Gz
                grad_shape[2,0:m] += np.interp(x=np.arange(t0,t1+dt,dt),
                                                xp=grad_list[u].get_timing(),
                                                fp=grad_list[u].get_shape()[:,2])

            # Case of arbitrary gradient
            elif isinstance(grad_list[u],blc.VaryingGradient):
                n = len(grad_list[u].get_shape)
                grad_shape[0][0:n] += grad_list[u].get_shape()[0] # Gx
                grad_shape[1][0:n] += grad_list[u].get_shape()[1] # Gy
                grad_shape[2][0:n] += grad_list[u].get_shape()[2] # Gz

        return blc.VaryingGradient(shape=grad_shape,timing=grad_timing)


# TODO : make main program work
# Main program here
if __name__ == '__main__':

    # Create phantom
    Nph = 5
    FOVph = 0.32
    Rs = [0.06, 0.12, 0.15]
    PDs = [1, 1, 1]
    T1s = [2, 1, 0.5]
    T2s = [0.1, 0.15, 0.25]
    phantom = pht.makeSphericalPhantom(n=Nph, fov=FOVph, T1s=T1s, T2s=T2s, PDs=PDs, radii=Rs)
    df = 0

    start_time = time.time()
    # Load pulseq file
    myseq = Sequence()
    myseq.read("gre_python_forsim_5.seq")

    loc_ind_list = phantom.get_list_inds()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(sim_single_sg, [(loc_ind, df, phantom, myseq) for loc_ind in loc_ind_list]).get()
    pool.close()


    my_signal = np.sum(results,axis=0)

    print("Time used: %s seconds" % (time.time()-start_time))

    np.save('pulseq_signal.npy', my_signal)

    # Display results
    ss = np.load('pulseq_signal.npy')
    plt.figure(3)
    plt.imshow(np.absolute(ss))
    plt.gray()

    aa = np.fft.fftshift(np.fft.ifft2(ss))
    plt.figure(4)
    plt.imshow(np.absolute(aa))
    plt.gray()
    plt.show()