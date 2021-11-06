import numpy as np
import matplotlib.pyplot as plt
from pypulseq.make_arbitrary_rf import make_arbitrary_rf

def sech(x):
    return 2/(np.exp(x)+np.exp(-x))

def tanh(x):
    return (np.exp(2*x) - 1)/(np.exp(2*x) + 1)

def make_hsi(a0, beta, mu, t, disp=False):
    # Make a hyperbolic secant pulse!
    a = a0*sech(beta*t)
    w1 = -mu*beta*tanh(beta*t)
    b1 = a*np.exp(-1j*w1*t)

    if disp:
        plt.subplot(311)
        plt.plot(t, a)
        plt.title('Amplitude modulation (AM)')

        plt.subplot(312)
        plt.plot(t, w1)
        plt.title('Frequency modulation (FM)')

        plt.subplot(313)
        plt.plot(t, np.real(b1))
        plt.plot(t, np.imag(b1))
        plt.legend(['Real', 'Imag'])
        plt.title('B1 waveform ')

        plt.show()

    return a, w1, b1

if __name__ == '__main__':
    a0 = 2
    beta = 800 # rad/s
    mu = 4.9
    Tp = 0.0072 # 7.2 ms
    Npts = 2000
    t = np.linspace(-Tp/2, Tp/2, Npts)

    A, W1, B1 = make_hsi(a0, beta, mu, t, disp=True)
