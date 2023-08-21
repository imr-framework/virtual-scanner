import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import virtualscanner.server.simulation.bloch.phantom as pht
from pypulseq.Sequence.sequence import Sequence
import time
import multiprocessing as mp
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim

if __name__ == '__main__':
    # Load seq file
    # Load the sequence : choose your own
    myseq = Sequence()
    myseq.read('DWI_pypulseq_colab_N16_Nb1_Ndir6_thk5.seq')
    # Diffusion sequence - b value (only one applied across.)
    seqb = loadmat('b_val.mat')['b'][0,0]  # From stored DWI info


    # Make custom cylindrical phantom (T1 plane, but with T1s set to be the same)
    T1T2PD0 = [1,0.5,1]  #
    PDs = [1,0.5,0.25]  # varying PD
    T1s = [1,1,1,1]
    T2s = [0.25, 0.12, 0.06, 0.02]
    T1T2PD1 = [1,0.5,1]
    myphantom = pht.makeCustomCylindricalPhantom(T1T2PD0, PDs, T1s, T2s, T1T2PD1, dim=2, n=16, dir='z', loc=0, fov=0.25)

    # Make diffusion map and add it to phantom
    tm = myphantom.type_map # 0->0, 4->2.4e-3(water), 5->0.6e-3(GM), 6->2.7e-3(CSF), 7->0.05e-3(lipids), 13->0
    Ddict = {0:0,4:2.4e-3,5:0.6e-3,6:2.7e-3,7:0.05e-3,13:0}
    Dmap = np.zeros(myphantom.type_map.shape)
    for u in Ddict.keys():
        Dmap[np.where(tm==u)] = Ddict[u]
    myphantom.Dmap = Dmap

    # Time the code: Tic
    start_time = time.time()
    # Store seq info
    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    # sg_type = 'Diffusion'
    sg_type = 'Default'


    df = 0
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                             [(loc_ind, df, myphantom, seq_info, sg_type, seqb) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('dwi_sim_try_default-sg.mat', {'signal': my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))