# Simulates received MR signal from any image (assuming uniform phase)
from matplotlib.image import imread
from math import pi
import matplotlib.pyplot as plt
import numpy as np



B0 = 3 # Tesla
GAMMA_BAR = 42.5775e6
w0 = B0*GAMMA_BAR*2*pi


bw_adc = 5000e3
dwell = 1/bw_adc
dt = 1e-9 # Sim.raster time dt = 1 ns (!?)

# plt.figure(1)
# plt.subplot(121)
# plt.imshow(coronal)
# plt.gray()
# plt.title('Image')
# plt.subplot(122)
# plt.imshow(np.log(np.absolute(kspace)))
# plt.title('k-space (log)')
# plt.gray()
# plt.show()


# 1. load image
myimage = imread('054.png')
# 2. Pad in 2nd dimension
Np, Nf = np.shape(myimage)
tmodel = np.arange(0,dwell*Nf,dt)
m = len(tmodel)
tmodel = tmodel[0:Nf+2*int((m-Nf)/2)]
# 3. 2D-FT
kspace = np.fft.fftshift(np.fft.fft2(myimage))
# 4. Modulate line by line
phi0 = 0
phi_B1 = 0
#cf = np.exp(1j*(w0*tmodel+phi_B1-phi0)) #
cf = np.sin(w0*tmodel+phi_B1-phi0)
klines = []
for p in range(Np): # For each "phase encode"
    kline = kspace[p,:]
    # zero fill to get
    imline = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(kline)))
    imline_zf = np.pad(imline,int((m-Nf)/2),'constant')
    kline_os = np.fft.fft(np.fft.fftshift(imline_zf))
    kline_mod = np.multiply(kline_os, cf)# modulate
    klines.append(kline_mod)


np.save('klines_b1ph0.npy',klines)

## Demodulation: multiply by cos/sin and then LPF
dw=0 # TODO Feel free to change this (demodulation frequency displacement)
dmf_rc = np.sin((w0+dw)*tmodel)
dmf_ic = np.cos((w0+dw)*tmodel)
myklines = np.load('klines_b1ph0.npy')
new_kspace = []
#for u in range(len(myklines)):
nn = round(dwell/dt)
nn = nn # TODO Fell free to change this and see effects of subsampling
for u in range(Np):
    # Multiply
    rc = np.multiply(myklines[u],dmf_rc)
    ic = np.multiply(myklines[u],dmf_ic)
    # LPF (simple rect filter in freq space)
    fvec = np.linspace(-1/(2*dt),1/(2*dt),len(rc))
    lpf = 1*(np.absolute(fvec)<(dw+w0)/(2*pi))
    rcf_lp = np.multiply(lpf,np.fft.fftshift(np.fft.fft(rc)))
    icf_lp = np.multiply(lpf,np.fft.fftshift(np.fft.fft(ic)))
    rc_dm = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(rcf_lp)))
    ic_dm = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(icf_lp)))
    # Sample
    rc_sampled = rc_dm[0::nn]
    ic_sampled = ic_dm[0::nn]
    # Store
    new_kspace.append(rc_sampled + 1j*ic_sampled)



new_kspace = np.array(new_kspace)
im_recon = (np.fft.ifft2(np.fft.fftshift(new_kspace)))

# Plot example signals (One readout line)
plt.figure(1)
plt.subplot(411)
plt.plot(np.absolute(kspace[-1,:]))
plt.title('Original k-space line')
plt.subplot(412)
plt.plot(myklines[-1])
plt.title('Modulated k-space line')
plt.subplot(413)
plt.plot(np.absolute(rc_dm + 1j*ic_dm))
plt.title('Demodulated k-space line')
plt.subplot(414)
plt.plot(np.absolute(new_kspace[-1,:]))
plt.title('Sampled demod. k-space line')

plt.figure(2)
plt.subplot(121)
plt.imshow(np.absolute(im_recon))
plt.gray()
plt.title('Recon')
plt.subplot(122)
plt.imshow(myimage)
plt.gray()
plt.title('Original')




