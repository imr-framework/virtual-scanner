# Converts the jemris simulation outputs  (signals.h5 files) into data or save as .npy or .mat files
# Gehua Tong
# March 06, 2020


import h5py
import numpy as np
import matplotlib.pyplot as plt

def recon_jemris(file, dims):
    Mxy_out, M_vec_out, times_out = read_jemris_output(file)
    kspace, imspace = recon_jemris_output(Mxy_out, dims)
    images = save_recon_images(imspace)# TODO save as png (use previous code!)

    return kspace, imspace, images


def read_jemris_output(file):
    # 1. Read simulated data
    f = h5py.File(file,'r')
    signal = f['signal']
    channels = signal['channels']

    # 2. Initialize output array
    Nch = len(channels.keys())
    Nro_tot = channels[list(channels.keys())[0]].shape[0]
    M_vec_out = np.zeros((Nro_tot,3,Nch))
    Mxy_out = np.zeros((Nro_tot,Nch), dtype=complex)
    times_out = np.array(signal['times'])

    # 3. Read each channel and store in array
    for ch, key in enumerate(list(channels.keys())):
        one_ch_data = np.array(channels[key])

        M_vec_out[:,:,ch] = one_ch_data
        Mxy_out[:,ch] = one_ch_data[:,0] + 1j*one_ch_data[:,1]


    return Mxy_out, M_vec_out, times_out


def recon_jemris_output(Mxy_out, dims):
    """Cartesian reconstruction of JEMRIS simulation output
    #  (No EPI/interleave reordering)

    Inputs
    ------
    Mxy_out : np.ndarray
        Complex Nt x Nch array where Nt is the total number of data points and Nch is the number of channels

    dims : array_like
        [Nro], [Nro, Nline], or [Nro, Nline, Nslice]

    """
    Nt, Nch = Mxy_out.shape
    print(Nt)
    if Nt != np.prod(dims):
        raise ValueError("The dimensions provided do not match the total number of samples.")
    Nro = dims[0]
    Nline = 1
    Nslice = 1

    ld = len(dims)

    if ld >= 1:
        Nro = dims[0]
    if ld >= 2:
        Nline = dims[1]
    if ld == 3:
        Nslice = dims[2]
    if ld > 3:
        raise ValueError("dims should have at 1-3 numbers : Nro, (Nline), and (Nslice)")

    kspace = np.zeros((Nro, Nline, Nslice, Nch),dtype=complex)
    imspace = np.zeros((Nro, Nline, Nslice, Nch),dtype=complex)

    np.reshape(Mxy_out, (Nro, Nline, Nslice))

    for ch in range(Nch):
        kspace[:,:,:,ch] = np.reshape(Mxy_out[:, ch], (Nro, Nline, Nslice), order='F')
        for sl in range(Nslice):
                imspace[:,:,sl,ch] = np.fft.fftshift(np.fft.ifft2(kspace[:,:,sl,ch]))

    return kspace, imspace


def save_recon_images(imspace, method='sum_squares'):
    if method == 'sum_squares':
        image = np.sum(np.square(np.absolute(imspace)),axis=-1)
    elif method == 'sum_abs':
        image = np.sum(np.absolute(imspace), axis=-1)
    else:
        raise ValueError("Method not recognized. Must be either sum_squares or sum_abs")
    return image


if __name__ == '__main__':
    Mxy_out, M_vec_out, times_out = read_jemris_output('sim/test0405/signals.h5')
    kk, im = recon_jemris_output(Mxy_out, dims=[15,15])
    images = save_recon_images(im)


    plt.figure(1)
    plt.subplot(121)
    plt.imshow(np.absolute(kk[:,:,0,0]))
    plt.title("k-space")
    plt.gray()
    plt.subplot(122)
    print(images)
    plt.imshow(np.squeeze(images[:,:,0]))
    plt.title("Image space")
    plt.gray()
    plt.show()