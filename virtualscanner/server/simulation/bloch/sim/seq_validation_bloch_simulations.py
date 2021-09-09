
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
    p = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=32)
   # p.loc = (0,0,0)
   # p.Zs = [0]

    #p = make_phantom_acr(N=128, FOV=0.25, slice_loc=0, shrink_factor=0.8, slice_type='grid')
    # Save phantom for repo.
    #T1map, T2map, PDmap, vsize, dBmap = 0, loc = (0, 0, 0)
    savemat('seq_validation_files/phantom_stored/T1plane.mat',
            {'T1map':p.T1map, 'T2map':p.T2map, 'PDmap': p.PDmap, 'vsize': p.vsize,
             'dBmap': p.dBmap, 'loc': p.loc})


    #myphantom = make_phantom_circle(N=32, FOV=0.25, slice_loc=0, shrink_factor=1)
    # 04/20 (line_thk = 1)
    #myphantom = make_phantom_acr(N=128, FOV=0.25, slice_loc=0, shrink_factor=0.8, slice_type='grid')
    #myphantom.output_h5(output_folder='seq_validation_files',name='pht0420_2')
    #myphantom.loc = (0,0,0)
    #myphantom.Zs = [0]
    #sp = {'circ_scale': 1, 'line_thk': N / 30, 'num_lines': 4}
    #myphantom = make_phantom_acr(N=128, FOV=0.25, slice_loc=0, slice_type='grid',slice_params=sp)
    #pht = make_phantom_acr(N=128, FOV=0.25, slice_loc=0, slice_type='grid')

    # Load the sequence : choose your own
    #myseq = Sequence()
    #myseq.read('seq_validation_files/irse64.seq')




    # Time the code: Tic
    #start_time = time.time()

    # Store seq info
    #seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    #loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    #pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    #sg_type = 'Solver'
    #results = pool.starmap_async(blcsim.sim_single_spingroup,
    #                             [(loc_ind, df, myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    #pool.close()
    # Add up signal across all SpinGroups
    #my_signal = np.sum(results, axis=0)
    #savemat('seq_validation_files/irse64_grid128.mat',{'signal':my_signal})

    # Time the code: Toc
    #print("Time used: %s seconds" % (time.time() - start_time))

