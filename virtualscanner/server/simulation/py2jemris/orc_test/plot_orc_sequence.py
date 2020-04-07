import h5py
import matplotlib.pyplot as plt

a = h5py.File('seq.h5')
print(a['seqdiag'].keys())
sd=a['seqdiag']

N1 = 20000
N2 = 30000

plt.figure(1)
plt.subplot(511)
plt.plot(sd['T'][N1:N2],sd['GX'][N1:N2])
plt.title('Gx')

plt.subplot(512)
plt.plot(sd['T'][N1:N2], sd['GY'][N1:N2])
plt.title('Gy')

plt.subplot(513)
plt.plot(sd['T'][N1:N2], sd['GZ'][N1:N2])
plt.title('Gz')

plt.subplot(514)
plt.plot(sd['T'][N1:N2], sd['TXM'][N1:N2])
plt.title('RF Magnitude')

plt.subplot(515)
plt.plot(sd['T'][N1:N2], sd['RXP'][N1:N2],'x')
plt.title('RXP')

plt.show()