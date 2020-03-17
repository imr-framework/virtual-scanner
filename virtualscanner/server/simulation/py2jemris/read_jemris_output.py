# Converts the jemris simulation outputs  (signals.h5 files) into data or save as .npy or .mat files
# Gehua Tong
# March 06, 2020


import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_jemris_output(file):
    # 1. Read simulated data
    f = h5py.File(file,'r')
    print(f.keys())
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
        print(one_ch_data.shape)
        M_vec_out[:,:,ch] = one_ch_data
        Mxy_out[:,ch] = one_ch_data[:,0] + 1j*one_ch_data[:,1]


    return Mxy_out, M_vec_out, times_out





if __name__ == '__main__':
    Mxy_out, M_vec_out, times_out = read_jemris_output('try_seq2xml/signals.h5')
    print(Mxy_out.shape)
    print(times_out.shape)

    Mxy_out = np.reshape(Mxy_out,(15,15))

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(np.absolute(Mxy_out))
    plt.title("k-space")
    plt.gray()
    plt.subplot(122)
    plt.imshow(np.absolute(np.fft.fftshift(np.fft.ifft2(Mxy_out))))
    plt.title("Image space")
    plt.gray()
    plt.show()