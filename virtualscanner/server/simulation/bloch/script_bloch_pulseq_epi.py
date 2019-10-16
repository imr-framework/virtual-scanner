# Copyright of the Board of Trustees of Columbia University in the City of New York
"""
Unit test for bloch simulation on pulseq EPI sequence (2D Cartesian, rectangular trajectory)
Run the script to generated a simulated image. Modify the code directly to set the phantom and acquisition parameters.
"""

import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

import virtualscanner.server.simulation.bloch.phantom as pht
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
import virtualscanner.server.simulation.bloch.pulseq_library as psl

if __name__ == '__main__':
    # Make a phantom
    nn = 21
    myphantom = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=nn)
    df = 0

    # Define sequence parameters
    # FOV = [0.24,0.24]
    FOV = 0.24
    # N = [21,21]
    N = 21
    FA = 90
    TR = 3
    TE = 0.1
    slice_locs = [0]
    thk = 0.24 / 21

    # Defining oblique encoding directions
    # Mrot = np.array([[1,            0,            0],
    #                  [0, np.cos(pi/4),-np.sin(pi/4)],
    #                  [0, np.sin(pi/4), np.cos(pi/4)]])
    # myenc = [[1,0,0],[0,1,0],[0,0,1]]
    # myenc[0] = np.transpose(Mrot@(np.transpose(myenc[0])))
    # myenc[1] = np.transpose(Mrot@(np.transpose(myenc[1])))
    # myenc[2] = np.transpose(Mrot@(np.transpose(myenc[2])))

    # Defining orthogonal encoding directions
    myenc = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    ns = 2
    myseq, ro_dirs, ro_order = psl.make_pulseq_epi_oblique(fov=FOV, n=N, thk=thk, fa=FA, tr=TR, te=TE, n_shots=ns,
                                                           enc=myenc, slice_locs=slice_locs, echo_type='se',
                                                           seg_type='interleaved', write=False)
    # Time the code: Tic
    start_time = time.time()

    # Store seq info
    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel CPU simulation
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df, myphantom, seq_info) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results, axis=0)
    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))

    # Multislice reconstruction
    Nf, Np = (N, N) if isinstance(N, int) else (N[0], N[1])
    Ns = len(slice_locs)
    im_mat = np.zeros((Nf, Np, Ns), dtype=complex)
    kspace = np.zeros((Nf, Np, Ns), dtype=complex)

    for v in range(Ns):  # For each slice
        slice_signal = my_signal[v * Np:v * Np + Np]
        # slice_signal[0::2] = np.fliplr(slice_signal[0::2]) # Reverses every other line because of EPI
        # Flip reversed readout lines
        for u in range(len(slice_signal)):
            if ro_dirs[u]:
                slice_signal[u] = np.flip(slice_signal[u])
        # Reorder readout lines (only for interleaved mode)
        if len(ro_order) != 0:
            #            print(np.round(slice_signal,2))
            #  print('reordering')
            slice_signal = slice_signal[ro_order]
        #  print(np.round(slice_signal,2))

        #        kspace[:, :, v] = np.transpose(my_signal[v * Np:v * Np + Np])
        kspace[:, :, v] = np.transpose(slice_signal)
        im_mat[:, :, v] = np.fft.fftshift(np.fft.ifft2(kspace[:, :, v]))

    sim_data = {}
    sim_data['kspace'] = kspace
    sim_data['image'] = im_mat
    np.save('pulseq_signal_new.npy', sim_data)

    # Display
    mydata = np.load('pulseq_signal_new.npy').all()
    image = mydata['image']
    kspace = mydata['kspace']
    Ns = np.shape(image)[2]
    a1 = int(np.sqrt(Ns))
    a2 = int(np.ceil(Ns / a1))

    plt.figure(1)
    for v in range(Ns):
        plt.subplot(a1, a2, v + 1)
        plt.imshow(np.absolute(kspace[:, :, v]))
        plt.gray()

    plt.figure(2)
    for v in range(Ns):
        plt.subplot(a1, a2, v + 1)
        plt.imshow(np.absolute(image[:, :, v]))
        plt.gray()

    plt.show()
