
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
    #myphantom = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=16)
    myphantom = pht.makePlanarPhantom(n=16,fov=0.25,T1s=[0.5],T2s=[0.05],PDs=[1],radii=[0.02],dir='z',loc=(0,0,0))
    #plt.imshow(myphantom.PDmap)
    #plt.show()
    savemat('minTE_test_point_phantom.mat',{'T1':myphantom.T1map, 'T2':myphantom.T2map, 'PD':myphantom.PDmap})
    # p.loc = (0,0,0)
    #  p.Zs = [0]


    # Load the sequence : choose your own
    myseq = Sequence()
    #myseq.read('ADC2_swift_FA5_N16.seq')
    myseq.read('ADC2_swift_FA5_N16.seq')

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
                                 [(loc_ind, df, myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    savemat('swift_signal.mat', {'signal':my_signal})

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))

    #########################################################

    # df = 0
    # #myphantom = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=16)
    # myphantom = pht.makePlanarPhantom(n=16,fov=0.25,T1s=[0.5],T2s=[0.05],PDs=[1],radii=[0.02],dir='z',loc=(0,0,0))
    # #plt.imshow(myphantom.PDmap)
    # #plt.show()
    # #savemat('minTE_test_point_phantom.mat',{'T1':myphantom.T1map, 'T2':myphantom.T2map, 'PD':myphantom.PDmap})
    # # p.loc = (0,0,0)
    # #  p.Zs = [0]
    #
    #
    # # Load the sequence : choose your own
    # myseq = Sequence()
    # myseq.read('code_TR100_TE0.16_FA15_N16.seq')
    #
    # # Time the code: Tic
    # start_time = time.time()
    # # Store seq info
    # seq_info = blcsim.store_pulseq_commands(myseq)
    # # Get list of locations from phantom
    # loc_ind_list = myphantom.get_list_inds()
    # # Initiate multiprocessing pool
    # pool = mp.Pool(mp.cpu_count())
    # # Parallel simulation
    # sg_type = 'Default'
    # results = pool.starmap_async(blcsim.sim_single_spingroup,
    #                              [(loc_ind, df, myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    # pool.close()
    # # Add up signal across all SpinGroups
    # my_signal = np.sum(results, axis=0)
    # savemat('code_signal.mat', {'signal':my_signal})
    #
    # # Time the code: Toc
    # print("Time used: %s seconds" % (time.time() - start_time))

