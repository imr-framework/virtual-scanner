
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
import matplotlib.pyplot as plt



if __name__ == '__main__':
    df = 0
    p = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=32, fov=0.25)
    #p.loc = (0,0,0)
    #p.Zs = [0]

    savemat('my_phantom.mat',{'T1':p.T1map, 'T2':p.T2map, 'PD':p.PDmap})


    # Load the sequence : choose your own
    myseq = Sequence()
    #myseq.read('ADC2_swift_FA5_N16.seq')
    myseq.read('spgr_sim_110221_no_adc_phase.seq')

    # Time the code: Tic
    start_time = time.time()
    # Store seq info
    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = p.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    sg_type = 'Default'
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df, p, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('signal_spgr_no_adc_phase_T1.mat', {'signal':my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))

