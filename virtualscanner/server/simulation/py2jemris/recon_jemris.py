# Converts the jemris simulation outputs  (signals.h5 files) into data or save as .npy or .mat files
# Gehua Tong
# March 06, 2020


import h5py
import numpy as np
import matplotlib.pyplot as plt

def recon_jemris(file, dims):
    """Reads JEMRIS's signals.h5 output, reconstructs it (Cartesian only for now) using the dimensions specified,
             and returns both the complex k-space and image matrix AND magnitude images

    Inputs
    ------
    file : str
        Path to signals.h5
    dims : array_like
        Dimensions for reconstruction
        [Nro], [Nro, Nline], or [Nro, Nline, Nslice]

    Returns
    -------
    kspace : np.ndarray
        Complex k-space
    imspace : np.ndarray
        Complex image space
    images : np.ndarray
        Real, channel-combined images


    """
    Mxy_out, M_vec_out, times_out = read_jemris_output(file)
    kspace, imspace = recon_jemris_output(Mxy_out, dims)
    images = save_recon_images(imspace)# TODO save as png (use previous code!)

    return kspace, imspace, images


def read_jemris_output(file):
    """Reads and parses JEMRIS's signals.h5 output

    Inputs
    ------
    file : str
        Path to signals.h5

    Returns
    -------
    Mxy_out : np.ndarray
        Complex representation of transverse magnetization sampled during readout
        Matrix dimensions : (total # readouts) x (# channels)
    M_vec_out : np.ndarray
        3D representation of magnetization vector (Mx, My, Mz) sampled during readout
        Matrix dimensions : (total # readouts) x 3 x (# channels)
    times_out : np.ndarray
        Timing vector for all readout points


    """
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


    Returns
    -------
    kspace : np.ndarray
        Complex k-space matrix
    imspace : np.ndarray
        Complex image space matrix

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
    """For now, this method combines channels and returns the image matrix
       (Future, for GUI use: add options to save as separate image files / mat / etc. in a directory)

    Inputs
    ------
    imspace : np.ndarray
        Complex image space. The last dimension must be # Channels.
    method : str, optional
        Method used for combining channels
        Either 'sum_squares' (default, sum of squares) or 'sum_abs' (sum of absolute values)

    Returns
    -------
    images : np.ndarray
        Real, channel_combined image matrix

    """
    if method == 'sum_squares':
        images = np.sum(np.square(np.absolute(imspace)),axis=-1)
    elif method == 'sum_abs':
        images = np.sum(np.absolute(imspace), axis=-1)
    else:
        raise ValueError("Method not recognized. Must be either sum_squares or sum_abs")
    return images


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