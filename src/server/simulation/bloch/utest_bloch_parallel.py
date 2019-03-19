import bloch as blc
import bloch_sequence as blcs
import blochsim as blsim
import phantom as pht
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def run_sim(simulator,loc,df):
    return simulator.apply_ps_to(loc,df)



# Run parallel simulation using mp.Pool
if __name__ == "__main__":
    kk = 9
    TR=0.5
    TE=0.05
    myseq = blcs.GRESequence(tr=TR, te=TE, flip_angle=np.pi / 2,
                             fov=[0.32, 0.32], num_pe=kk, num_fe=kk, slice_locs=[0], thk=0.32 / kk,
                             gmax=10e-3, srmax=500)
    # Make phantom
    Nph = kk
    FOVph = 0.32
    Rs = [0.06, 0.12, 0.15]
    PDs = [1, 1, 1]
    T1s = [2, 1, 0.5]
    T2s = [0.1, 0.15, 0.25]
    myphantom = pht.makeSphericalPhantom(n=Nph, fov=FOVph, T1s=T1s, T2s=T2s, PDs=PDs, radii=Rs)

    loc_ind_list = myphantom.get_list_inds()
    mysim = blsim.BlochSimulator(myphantom, myseq)
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(run_sim, [(mysim,ind,0) for ind in loc_ind_list]).get()
    pool.close()

    # Sum all isochromats :) ~
    my_signal = np.sum(results,axis=0)
    np.save('my_signal_p2.npy', my_signal)

    # Calculate image with signal equation
    t1= myphantom.get_T1map()[:,:,int(kk/2)]
    t2= myphantom.get_T2map()[:,:,int(kk/2)]
    pd= myphantom.get_PDmap()[:,:,int(kk/2)]
    bb = np.multiply(1-np.exp(-np.divide(TR,t1)),np.exp(-np.divide(TE,t2)))
    bb = np.multiply(bb,pd)

    # recon
    # 2D IFFT and visualize
    ss = np.load('my_signal_p2.npy')
    plt.figure(1)
    plt.subplot(121)
    aa = np.absolute(np.fft.fftshift(np.fft.ifft2(ss)))
    plt.imshow(aa)
    plt.gray()
    plt.title("Reconstructed image")
    plt.subplot(122)
    plt.imshow(bb)
    plt.gray()
    plt.title("Signal equation prediction")


    plt.figure(2)

    plt.subplot(121)
    plt.imshow(np.absolute(ss))
    plt.gray()
    plt.title("Simulated k-space")

    plt.subplot(122)
    plt.title("Signal equation k-space")
    plt.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(bb))))
    plt.show()


