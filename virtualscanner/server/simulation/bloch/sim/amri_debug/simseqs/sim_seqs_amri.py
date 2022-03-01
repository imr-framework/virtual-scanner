
import multiprocessing as mp
import time

#import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from pypulseq.Sequence.sequence import Sequence
import virtualscanner.server.simulation.bloch.phantom as pht
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
import virtualscanner.server.simulation.bloch.pulseq_library as psl
from virtualscanner.server.simulation.bloch.phantom_acr import make_phantom_acr, make_phantom_circle

if __name__ == '__main__':
    df = 0
    myphantom = pht.makeCylindricalPhantom(dim=2, dir='z', loc=-0.08, n=32, fov=0.24) # Grab T2 plane
    myphantom.loc = (0,0,0)
    myphantom.Zs = [0]


    # Load the sequence : choose your own
    myseq = Sequence()
    myseq.read('flair_new/flair_fast_ms_FOV240mm_N32_Ns1_TR10000ms_TE13ms_TI2600ms.seq')


    ########################################################################################
    # Time the code: Tic
    start_time = time.time()
    # Store seq info
    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    sg_type = 'Solver'
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df, myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('flair_new/flair_new_signal.mat', {'signal':my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))

    ##############################################################################################