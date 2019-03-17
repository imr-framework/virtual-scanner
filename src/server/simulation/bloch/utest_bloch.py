"""
 Unit testing for bloch simulation module
# 03/17/2019
# Gehua Tong
"""
import bloch as blc
import bloch_sequence as bseq
import blochsim as bsim
import phantom as pht
import numpy as np
import matplotlib.pyplot as plt



# Construct a spherical phantom
Nph = 5
FOVph = 0.32
Rs = [0.06,0.12,0.15]
PDs = [1,1,1]
T1s = [2,1,0.5]
T2s = [0.1,0.15,0.25]

myphantom = pht.makeSphericalPhantom(n=Nph,fov=FOVph,T1s=T1s,T2s=T2s,PDs=PDs,radii=Rs)


# Create a GRE sequence with 3 slices
fov_fe = 0.32
fov_pe = 0.32
TR = 0.5
TE = 0.05
myseq = bseq.GRESequence(tr=TR,te=TE,flip_angle=np.pi/2,
                         fov=[fov_fe,fov_pe],num_pe=5,num_fe=5,slice_locs=[0],thk=0.32/5,
                         gmax=10e-3,srmax=500)


# Initialize a simulator object and simulate
mysim = bsim.BlochSimulator(myphantom,myseq)
mysim.simulate()
s = mysim.get_signal()
np.save('my_signal.npy',s)


# Reconstruct signal and display results
ss = np.load('my_signal.npy')
plt.figure(1)
plt.imshow(np.absolute(np.fft.fftshift(np.fft.ifft2(ss))))
plt.gray()
plt.show()
plt.title("Reconstructed image")



