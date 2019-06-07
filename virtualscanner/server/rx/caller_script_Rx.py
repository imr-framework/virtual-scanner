# Simulates received MR signal from any image (assuming uniform phase)
from matplotlib.image import imread
from math import pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def run_Rx_sim(Rxinfo):
    B0 = 3  # Tesla
    GAMMA_BAR = 42.5775e6
    w0 = B0 * GAMMA_BAR * 2 * pi
    bw_adc = 5000e3
    dwell = 1 / bw_adc
    dt = 1e-9  # Sim.raster time dt = 1 ns

    # Read parameters # TODO change according to payload format
    dw = 2*pi*float(Rxinfo['deltaf'])
    im_path = './src/server/rx/'+ Rxinfo['image-or'] + '.png'
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
        new_kspace_with_zeros = np.zeros((np.shape(new_kspace)[0], dsf * np.shape(new_kspace)[1]),dtype=complex)
        new_kspace_with_zeros[:, 0::dsf] = new_kspace
        new_kspace = new_kspace_with_zeros

    timestamp = time.strftime("%Y%m%d%H%M%S")
    im_recon = (np.fft.ifft2(np.fft.fftshift(new_kspace)))

    # Save images
    mypath = './src/coms/coms_ui/static/rx/outputs/'
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    # Plot signals at different stages & save
    #ind = round(Np/2)
    ind = -1
    plt.figure(1)
    plt.tight_layout()
    plt.subplot(411)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.absolute(kspace[ind,:]))
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
    plt.plot(np.absolute(new_kspace[ind,:]))
    plt.title('Sampled demod. k-space line')
    plt.tight_layout()
    signals_plot_path = mypath + 'Rx_signals_'+timestamp+'.png'
    plt.savefig(signals_plot_path,bbox_inches='tight',pad_inches=0,format='png')
    plt.clf()

    # Recon image from rx signal
    plt.figure(2)
    plt.axis("off")
    fig = plt.imshow(np.absolute(im_recon))
    plt.gray()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    recon_im_path = mypath + 'recon_im_'+timestamp+'.png'
    plt.savefig(recon_im_path,bbox_inches='tight',pad_inches=0,format='png')
    plt.clf()

    return  [signals_plot_path, recon_im_path, im_path]



if __name__ == "__main__":
    Rxinfo = {'image-or':'coronal','DSF':1, 'deltaf':0}
    a = run_Rx_sim(Rxinfo)
    print(a)
    b = imread(a[2])