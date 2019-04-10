# Unit test for direct-from-pulseq bloch simulation

import blochsim_pulseq as blcsim
import numpy as np
import matplotlib.pyplot as plt
import time
import bloch as blc
import phantom as pht
from pulseq.core.Sequence.sequence import Sequence
from pulseq.core.Sequence.read_seq import read
import multiprocessing as mp
import spingroup_ps as sg
import scipy.io as sio
import pulseq_library as psl

if __name__ == '__main__':
    # Make a phantom
    Nph = 15
    FOVph = 0.32 #(for GRE sequence with FOV = 0.32)
    Rs = [0.06,0.12,0.15] #(for GRE sequence with FOV = 0.32)
    PDs = [1, 1, 1]
    T1s = [2, 1, 0.5]
    T2s = [0.1, 0.15, 0.25]

    #phantom = pht.makeSphericalPhantom(n=Nph, fov=FOVph, T1s=T1s, T2s=T2s, PDs=PDs, radii=Rs)
    phantom = pht.makePlanarPhantom(n=Nph, fov=FOVph, T1s=T1s, T2s=T2s, PDs=PDs, radii=Rs)
    df = 0

    # Make a pulse sequence
    fov = FOVph
    n = Nph
    thk = FOVph/n
    fa = 90
    tr = 10
    te = 0.08
    ti = [0.02]

    #myseq = psl.make_pulseq_gre(fov,n,thk,fa,tr,te,write=False)
    myseq = psl.make_pulseq_irse(fov,n,thk,fa,tr,te,ti,write=False)


    # Time the code: Tic
    start_time = time.time()
    loc_ind_list = phantom.get_list_inds()

    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    # Multiprocessing simulation!
    results = pool.starmap_async(blcsim.sim_single_spingroup, [(loc_ind, df, phantom, myseq) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results,axis=0)

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time()-start_time))

    # Save signal
    np.save('pulseq_signal_new.npy', my_signal)

    # Reload signal and display results
    ss = np.load('pulseq_signal_new.npy')
    plt.figure(1)
    plt.imshow(np.absolute(ss))
    plt.gray()
    plt.title('K-space')

    aa = np.fft.fftshift(np.fft.ifft2(ss))
    plt.figure(2)
    plt.imshow(np.absolute(aa))
    plt.title('Image space')
    plt.gray()
    plt.show()

