# Copyright of the Board of Trustees of Columbia University in the City of New York
# For conventionally 2D encoded (i.e. Cartesian; one readout line per phase encode) pulseq sequences
"""
Unit test for bloch simulation on GRE, SE, and IRSE sequences (2D Cartesian, line-by-line, rectangular trajectory)
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
    nn = 15
    myphantom = pht.makeCylindricalPhantom(dim=2,dir='z',loc=0,n=nn)

    df = 0
    FOV = [0.24,0.24]
    N = [15,15]
    FA = 90
    TR = 5
    TE = 0.10
    TI = 0.10
    slice_locs = [0]
    thk = 0.3/15

    # Defining oblique encoding directions
    #Mrot = np.array([[1,0,0],
    #                 [0,np.cos(pi/4),-np.sin(pi/4)],
    #                [0,np.sin(pi/4),np.cos(pi/4)]])
    #myenc = [[1,0,0],[0,1,0],[0,0,1]]
    #myenc[0] = np.transpose(Mrot@(np.transpose(myenc[0])))
    #myenc[1] = np.transpose(Mrot@(np.transpose(myenc[1])))
    #myenc[2] = np.transpose(Mrot@(np.transpose(myenc[2])))

    # Defining orthogonal encoding directions
    myenc = [(1,0,0),(0,1,0),(0,0,1)]


    # Make the sequence : choose your own
    # myseq = psl.make_pulseq_irse(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,ti=TI,enc='xyz',slice_locs=slice_locs,write=False)
    # myseq = psl.make_pulseq_se(fov=FOV,n=N,thk=FOV/N,fa=FA,tr=TR,te=TE,enc='xyz',slice_locs=[0],write=False)
    # myseq = psl.make_pulseq_gre_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=myenc,slice_locs=[0],write=False)
    # myseq = psl.make_pulseq_irse_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,ti=TI,enc=myenc,slice_locs=[0],write=False)
    myseq = psl.make_pulseq_se_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=myenc,slice_locs=[0],write=False)
    myseq.write('oblique.seq')
    myseq.read('oblique.seq')

    # Time the code: Tic
    start_time = time.time()

    # Store seq info
    seq_info = blcsim.store_pulseq_commands(myseq)
    # Get list of locations from phantom
    loc_ind_list = myphantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    results = pool.starmap_async(blcsim.sim_single_spingroup, [(loc_ind, df, myphantom, seq_info) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    my_signal = np.sum(results,axis=0)

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time()-start_time))


    # Multislice reconstruction
    Nf, Np = (N,N) if isinstance(N,int) else (N[0],N[1])
    Ns = len(slice_locs)
    im_mat = np.zeros((Nf, Np, Ns), dtype=complex)
    kspace = np.zeros((Nf, Np, Ns), dtype=complex)
    for v in range(Ns):
        kspace[:, :, v] = np.transpose(my_signal[v * Np:v * Np + Np])
        im_mat[:, :, v] = np.fft.fftshift(np.fft.ifft2(kspace[:, :, v]))
    sim_data = {}
    sim_data['kspace'] = kspace
    sim_data['image'] = im_mat
    np.save('pulseq_signal_new.npy', sim_data)

    # Display
    mydata = np.load('pulseq_signal_new.npy', allow_pickle=True).all()
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

