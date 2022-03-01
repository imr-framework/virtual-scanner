
import multiprocessing as mp
import time
import virtualscanner.server.simulation.bloch.spingroup_ps as sg
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from pypulseq.Sequence.sequence import Sequence
import virtualscanner.server.simulation.bloch.phantom as pht
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
import virtualscanner.server.simulation.bloch.pulseq_library as psl
from virtualscanner.server.simulation.bloch.phantom_acr import make_phantom_acr, make_phantom_circle

if __name__ == '__main__':
    seq = Sequence()
    seq.read('spgr_gspoil_debug.seq')


    #
    # seq_info = blcsim.store_pulseq_commands(seq)
    # isc = sg.SpinGroup(loc=(-0.1,0.1,0), pdt1t2=(1,0.5,0.05), df=0)
    # m_store = blcsim.apply_pulseq_commands(isc,seq_info,store_m=False)
    # #print(m_store)
    # savemat('stored_sim_magnetizations.mat',{'m_store':m_store, 'seq_info':seq_info, 'signal': np.array(isc.signal)})

    phantom_type = 't1_plane'
    # Phantom matrix size (isotropic)
    Npht = 16
    if phantom_type == 't1_plane':
      p = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=Npht)
    elif phantom_type == 't2_plane':
      p = pht.makeCylindricalPhantom(dim=2, dir='z', loc=-0.08, n=Npht)
      p.loc = (0,0,0)
      p.Zs = [0]
    elif phantom_type == 'pd_plane':
      p = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0.08, n=Npht)
      p.loc = (0,0,0)
      p.Zs = [0]

    myphantom = p
    # Run simulation
    # Time the code: Tic
    start_time = time.time()
    # Store seq info
    seq_info = blcsim.store_pulseq_commands(seq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    sg_type = 'Solver'
    df = 0 # no off-resonance (B0) map
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df, myphantom, seq_info,sg_type) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    signal = np.sum(results, axis=0)
    # Save data (you can download this from the Files sidebar on the left)
    savemat('simulated_signal.mat',{'signal':signal})
    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))


    kk = signal
    # 2D IFFT
    images = np.absolute(np.fft.fftshift(np.fft.ifft2(kk)))
    plt.imshow(np.absolute(images))
    plt.gray()
    plt.show()