from virtualscanner.server.simulation.bloch.spingroup_ps import SpinGroup
import numpy as np
# TODO : this class!
# Uses the same form of simulation, but actually splits into 10 (or n) spins with
# freq. deviations drawn from the prob. distribution outlined in JEMRIS paper.
# This should enable a SE-refocusable phase
# Can we do just one spin?
# Do we require location drift?How? 2D/3D Gaussian random?
# Test with SE vs. GRE, circle phantom.

class SpinGroupT2star(SpinGroup):
    def __init__(self, loc=(0,0,0), pdt1t2=(1,0,0), t2star=0, num_spins=10):
        super.__init__(loc=loc,pdt1t2=pdt1t2,df=None)
        # Diffusion coefficient in [(mm^2)/seconds]
        self.num_spins = num_spins
        self.T2star = t2star

        if t2star == 0:
            self.t2p = 0
        elif t2star < 0:
            raise ValueError("T2 star must not be negative. ")
        elif t2star > self.T2:
            raise ValueError("T2 star must not be longer than T2. ")

        self.T2prime = 1/(1/t2star - 1/self.T2)

        # Initialize list of spins
        self.spin_list = [SpinGroup() for i in range(num_spins)]
        # Assign random frequencies using the
        #        inverse Cauchy-Lorentz cumuluative distribution
        for spin in self.spin_list:
            spin.df = (1/self.T2prime) * np.tan(np.pi*(np.random.rand() - 0.5))/(2*np.pi)

    # Methods the same as SpinGroup:
    def scale_m_signal(self, scale):
        for spin in self.spin_list: spin.scale_m_signal(scale)

    def get_m_signal(self):
       # Get m signal - an average of all num_spins spins!
       m_signal = 0
       for spin in self.spin_list:
            m_signal += spin.get_m_signal()

       m_signal /= self.num_spins
       return m_signal

    def get_magnetizations_all(self):
        return [spin.m for spin in self.spin_list]

    # Query method (new)
    def get_avg_signal(self):
        avg_signal = np.zeros(self.spin_list[0].signal.shape, dtype=complex)
        for spin in self.spin_list:
            avg_signal += np.array(spin.signal)
        avg_signal /= self.num_spins
        return avg_signal

    # Query method (new)
    def fpwg(self,grad_area,t):
        for spin in self.spin_list: spin.fpwg(grad_area, t)

    def delay(self,t):
        for spin in self.spin_list: spin.delay(t)

    def apply_rf(self, pulse_shape, grads_shape, dt):
        for spin in self.spin_list: spin.apply_rf(pulse_shape, grads_shape, dt)

    def apply_rf_store(self, pulse_shape, grads_shape, dt):
        m_signal_total = np.zeros(len(pulse_shape), dtype=complex)
        magnetizations_total = np.zeros((3, len(pulse_shape) + 1))
        for spin in self.spin_list:
            m_signal, magnetizations = spin.apply_rf_store(pulse_shape, grads_shape, dt)
            m_signal_total += m_signal
            magnetizations_total += magnetizations
        m_signal_avg = m_signal_total / self.num_spins
        magnetizations_avg = magnetizations_total / self.num_spins
        return m_signal_avg, magnetizations_avg

    def readout_trapz(self,dwell,n,delay,grad,timing,phase):
        for spin in self.spin_list: spin.readout_trapz(dwell,n,delay,grad,timing,phase)

    def readout(self,dwell,n,delay,grad,timing,phase):
        for spin in self.spin_list: spin.readout_trapz(dwell,n,delay,grad,timing,phase)
