# Unit test for direct-from-pulseq bloch simulation

import pulseq_blochsim_methods as blcsim
import numpy as np
import matplotlib.pyplot as plt
import time
import phantom as pht
import multiprocessing as mp
import pulseq_library as psl
from pypulseq.Sequence.sequence import Sequence


if __name__ == '__main__':

    # Make a phantom
    #Nph = 15
    #FOVph = 0.32 #(for GRE sequence with FOV = 0.32)
    #Rs = [0.06,0.12,0.15] #(for GRE sequence with FOV = 0.32)
    #PDs = [1, 1, 1]
    #T1s = [2, 1, 0.5]
    #T2s = [0.1, 0.15, 0.25]
    nn = 15
    myphantom = pht.makeCylindricalPhantom(dim=2,n=nn,dir='z',loc=0)
#    myphantom = pht.makePlanarPhantom(n=Nph,fov=FOVph,T1s=T1s,T2s=T2s,PDs=PDs,radii=Rs)


    df = 0 # TODO add intra-spingroup dephasing - make it part of spingroup_ps!

    FOV = 0.24
    N = nn
    FA = 90
    TR = 2
    TE = 0.10
    TI = 0.10
    myseq = psl.make_pulseq_se(fov=FOV,n=N,thk=FOV/N,fa=FA,tr=TR,te=TE,enc='xyz',slice_locs=[0],write=False)

 #   myseq = psl.make_pulseq_se(fov=FOV,n=N,thk=FOV/N,fa=FA,tr=TR,te=TE,enc='xyz',slice_locs=[0],write=False)
    #myseq = Sequence()
    #myseq.read('se_python_forsim_15.seq')

    # Time the code: Tic
    start_time = time.time()

    # Store seq info
#    seq_info = blcsim.store_pulseq_commands(myseq)

    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()


    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    # Multiprocessing simulation!
    results = pool.starmap_async(blcsim.sim_single_spingroup, [(loc_ind, df, myphantom, seq_info) for loc_ind in loc_ind_list]).get()
    #results = pool.starmap_async(blcsim.sim_single_spingroup_new, [(loc_ind, df, myphantom, seq_info) for loc_ind in loc_ind_list]).get()

    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results,axis=0)

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time()-start_time))

    # Save signal
    np.save('pulseq_signal_new.npy', my_signal)

    #####################################
    # Reload signal and display results #
    #####################################

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

