# Copyright of the Board of Trustees of Columbia University in the City of New York

from math import pi

import matplotlib
from matplotlib.image import imread

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from virtualscanner.utils import constants

COMS_RX_INPUTS_PATH = constants.COMS_UI_STATIC_RX_INPUT_PATH
COMS_RX_OUTPUTS_PATH = constants.COMS_UI_STATIC_RX_OUTPUT_PATH


def run_Rx_sim(Rxinfo):
    """
    Runs MR receive chain simulation

    Receive chain is simulated by modulating k-space from an axial, coronal, or sagittal brain image,
    and then demodulating using selected demod. frequency and sampling at selected down-sampling factor (dsf).

    Parameters
    ----------
    Rxinfo : dict
        | Dictionary of parameters used to run Rx simulation
        | {'deltaf': demod_frequency, 'image-or': image_orientation, 'DSF': downsampling_factor}

    Returns
    -------
    signals_plot_filename : str
        File name for plot of time-domain signals (.png format)
    recon_plot_filename : str
        File name for image reconstructed from demodulated & sampled signals (.png format)
    im_path : str
        Path of image used for generating artificial k-space
    """
    B0 = 3  # Tesla
    GAMMA_BAR = 42.5775e6
    w0 = B0 * GAMMA_BAR * 2 * pi
    bw_adc = 5000e3
    dwell = 1 / bw_adc
    dt = 1e-9  # Sim.raster time dt = 1 ns

    # Read parameters
    dw = 2 * pi * float(Rxinfo['deltaf'])
    im_path = str(COMS_RX_INPUTS_PATH) + '/' + Rxinfo['image-or'] + '.png'
    dsf = int(Rxinfo['DSF'])

    # 1. load image
    myimage = imread(im_path)
    # 2. Pad in 2nd dimension
    Np, Nf = np.shape(myimage)
    tmodel = np.arange(0, dwell * Nf, dt)
    m = len(tmodel)
    tmodel = tmodel[0:Nf + 2 * int((m - Nf) / 2)]
    # 3. 2D-FT
    kspace = np.fft.fftshift(np.fft.fft2(myimage))
    # 4. Modulate line by line
    phi0 = 0
    phi_B1 = 0
    cf = np.sin(w0 * tmodel + phi_B1 - phi0)

    klines = []

    for p in range(Np):  # For each "phase encode"
        kline = kspace[p, :]
        # zero fill to get
        imline = np.fft.fftshift(np.fft.ifft(kline))
        imline_zf = np.pad(imline, int((m - Nf) / 2), 'constant')
        kline_os = np.fft.fft(np.fft.fftshift(imline_zf))

        kline_mod = np.multiply(kline_os, cf)  # modulate
        klines.append(kline_mod)

    ## Demodulation: multiply by cos/sin and then LPF
    dmf_rc = np.sin((w0 + dw) * tmodel)
    dmf_ic = np.cos((w0 + dw) * tmodel)
    new_kspace = []
    nn = round(dwell / dt) * dsf
    rc_dm = []
    ic_dm = []

    for u in range(Np):
        # Multiply
        rc = np.multiply(klines[u], dmf_rc)
        ic = np.multiply(klines[u], dmf_ic)
        # LPF (simple rect filter in freq space)
        fvec = np.linspace(-1 / (2 * dt), 1 / (2 * dt), len(rc))
        lpf = 1 * (np.absolute(fvec) < (dw + w0) / (2 * pi))
        # Low pass filter in freq domain
        rcf_lp = np.multiply(lpf, np.fft.fftshift(np.fft.fft(rc)))
        icf_lp = np.multiply(lpf, np.fft.fftshift(np.fft.fft(ic)))
        # Back to time domain
        rc_dm.append((np.fft.ifft(np.fft.fftshift(rcf_lp))))
        ic_dm.append((np.fft.ifft(np.fft.fftshift(icf_lp))))
        # Sample
        rc_sampled = rc_dm[u][0::nn]
        ic_sampled = ic_dm[u][0::nn]
        # Store
        new_kspace.append(rc_sampled + 1j * ic_sampled)

    new_kspace = np.array(new_kspace)

    # Zero insertion for downsampled cases
    if dsf > 1:
        new_kspace_with_zeros = np.zeros((np.shape(new_kspace)[0], dsf * np.shape(new_kspace)[1]), dtype=complex)
        new_kspace_with_zeros[:, 0::dsf] = new_kspace
        new_kspace = new_kspace_with_zeros

    timestamp = time.strftime("%Y%m%d%H%M%S")
    im_recon = (np.fft.ifft2(np.fft.fftshift(new_kspace)))

    # Save images
    rx_outputs_path = str(COMS_RX_OUTPUTS_PATH) + '/'
    if not os.path.isdir(rx_outputs_path):
        os.makedirs(rx_outputs_path)

    # Plot signals at different stages & save
    # ind = round(Np/2)
    ind = -1  # using the last line
    plt.figure(1)
    plt.tight_layout()
    plt.subplot(411)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.absolute(kspace[ind, :]))
    plt.title('Original k-space line')
    plt.subplot(412)
    plt.xticks([])
    plt.yticks([])
    plt.plot(klines[ind])
    plt.title('Modulated k-space line')
    plt.subplot(413)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.absolute(rc_dm[ind] + 1j * ic_dm[ind]))
    plt.title('Demodulated k-space line')
    plt.subplot(414)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.absolute(new_kspace[ind, :]))
    plt.title('Sampled demod. k-space line')
    plt.tight_layout()

    signals_plot_filename = 'Rx_signals_' + timestamp + '.png'
    signals_plot_path = rx_outputs_path + signals_plot_filename
    plt.savefig(signals_plot_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.clf()

    # Recon image from rx signal
    plt.figure(2)
    plt.axis("off")
    fig = plt.imshow(np.absolute(im_recon))
    plt.gray()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    recon_plot_filename = 'recon_im_' + timestamp + '.png'
    recon_im_path = rx_outputs_path + recon_plot_filename
    plt.savefig(recon_im_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.clf()

    return signals_plot_filename, recon_plot_filename, im_path

if __name__ == "__main__":
   Rxinfo = {'image-or':'coronal','DSF':1, 'deltaf':0}
   signals_filename,recon_filename,im_path = run_Rx_sim(Rxinfo)
   print('Signal plot is stored in: ' + signals_filename)
   print('Reconstructed image is stored in: ' + recon_filename)

