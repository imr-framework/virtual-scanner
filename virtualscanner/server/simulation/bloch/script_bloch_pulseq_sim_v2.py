import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import virtualscanner.server.simulation.bloch.phantom as pht
from pypulseq.Sequence.sequence import Sequence
import time
import multiprocessing as mp
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim


if __name__ == '__main__':
    # Make phantom

    dfmap = np.zeros((32,32,1))
    txmap = np.ones((32,32,1))
    rxmap = np.ones((32,32,1))


    myphantom = pht.makeCylindricalPhantom(dim=2,n=32,dir='z',loc=0,fov=0.25)

    # Load the sequence : choose your own
    myseq = Sequence()
    myseq.read('sim/amri_debug/b0demo/tse_ms_FOV250mm_N32_Ns1_TR2000ms_TE12ms_4echoes_dwell200.0us.seq')

    # Time the code: Tic
    start_time = time.time()
    # Store seq info
    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    sg_type = 'Default'


    scanner_info = {'B0': dfmap, 'B1tx':txmap, 'B1rx':rxmap}


    results = pool.starmap_async(blcsim.sim_single_spingroup_v2,
                                 [(loc_ind, myphantom, seq_info, scanner_info, sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('db0_sim_signal_t1plane_tse_testv2.mat', {'signal':my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))