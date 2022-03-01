import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import virtualscanner.server.simulation.bloch.phantom as pht
from pypulseq.Sequence.sequence import Sequence
import time
import multiprocessing as mp
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
if __name__ == '__main__':
    myphantom = pht.makeCylindricalPhantom(dim=2,n=16,dir='z',loc=0,fov=0.25)
    # Load the sequence : choose your own
    myseq = Sequence()
    myseq.read('spgr_rfspoil_mmj.seq')

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
    df = 0
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df, myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('spgr_rfspoil_adc_phase_test.mat', {'signal':my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))