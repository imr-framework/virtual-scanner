import h5py
import matplotlib.pyplot as plt
import numpy as np


sp1 = h5py.File('sim/test0501/gre_test_0417.h5','r')
sp2 = h5py.File('sim/test0501/g','r')

print(sp1['seqdiag'].keys())
print(sp2['seqdiag'].keys())


name = 'RXP'

plt.figure(1)
plt.subplot(211)
plt.title(name + ' (original)')
plt.plot(sp1['seqdiag/T'], sp1['seqdiag' + f'/{name}'],'*')
plt.subplot(212)
plt.title(name + ' (twice)')
plt.plot(sp2['seqdiag/T'], sp2['seqdiag' + f'/{name}'],'*')

plt.show()


rxp_1 = np.array(sp1['seqdiag/RXP'])
rxp_2 = np.array(sp2['seqdiag/RXP'])

print(rxp_1*180/np.pi)
print(rxp_2*180/np.pi)