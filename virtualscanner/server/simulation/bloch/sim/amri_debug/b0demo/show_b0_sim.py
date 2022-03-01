import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import virtualscanner.server.simulation.bloch.phantom as pht
from pypulseq.Sequence.sequence import Sequence
import time
import multiprocessing as mp
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
if __name__ == '__main__':
    # Load interpolated B0 map and make brainweb phantom!
    bwdata = loadmat('brainweb_sim_source_data_32.mat')
    #typemap = bwdata['bwslice32']
    b0map = bwdata['b0slice32']
    df = np.zeros((32,32,1))
    #df[:,:,0] = b0map
    #
    # typemap_3d = np.zeros((32,32,1))
    # typemap_3d[:,:,0] = typemap
    #
    # # Make brainweb phantom
    # FOV = 0.25
    # N = 32
    # params_mat = loadmat('bw_params.mat')['params']
    # print(params_mat.shape)
    # typedict = {}
    # for u in range(params_mat.shape[0]):
    #     typedict[u] = (params_mat[u,3],params_mat[u,0]/1e3,params_mat[u,1]/1e3)
    #
    # myphantom = pht.DTTPhantom(type_map=typemap_3d, type_params=typedict, vsize=FOV/N, dBmap=0, loc=(0,0,0))
    #



    myphantom = pht.makeCylindricalPhantom(dim=2,n=32,dir='z',loc=0,fov=0.25)

    # Load the sequence : choose your own
    myseq = Sequence()
    myseq.read('tse_ms_FOV250mm_N32_Ns1_TR2000ms_TE12ms_4echoes_dwell200.0us.seq')

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
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df[loc_ind], myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('db0_sim_signal_t1plane_tse_nodf.mat', {'signal':my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))