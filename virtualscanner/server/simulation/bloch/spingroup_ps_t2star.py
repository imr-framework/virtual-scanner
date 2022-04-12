from virtualscanner.server.simulation.bloch.spingroup_ps import SpinGroup, NumSolverSpinGroup
import numpy as np
import time
from pypulseq.opts import Opts
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.calc_duration import calc_duration
from pypulseq.make_delay import make_delay
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
from scipy.io import savemat, loadmat

# TODO : this class!
# Uses the same form of simulation, but actually splits into 10 (or n) spins with
# freq. deviations drawn from the prob. distribution outlined in JEMRIS paper.
# This should enable a SE-refocusable phase
# Can we do just one spin?
# Do we require location drift?How? 2D/3D Gaussian random?
# Test with SE vs. GRE, circle phantom.

class SpinGroupT2star(NumSolverSpinGroup):
    def __init__(self, loc=(0,0,0), pdt1t2=(1,0,0), df=0, t2star=0, num_spins=10):
        super().__init__(loc=loc,pdt1t2=pdt1t2,df=df)
        # Diffusion coefficient in [(mm^2)/seconds]
        self.num_spins = num_spins
        self.T2star = t2star

        if t2star == 0:
            self.T2prime = 0
        elif t2star < 0:
            raise ValueError("T2 star must not be negative. ")
        elif t2star > self.T2:
            raise ValueError("T2 star must not be longer than T2. ")
        else:
            self.T2prime = 1/(1/t2star - 1/self.T2)

        # Initialize list of spins
        self.spin_list = [NumSolverSpinGroup(loc=loc, pdt1t2=pdt1t2,df=df) for i in range(num_spins)]
        # Assign random frequencies using the
        #        inverse Cauchy-Lorentz cumuluative distribution
        if self.T2prime > 0:
            for spin in self.spin_list:
                spin.df += (1/self.T2prime) * np.tan(np.pi*(np.random.rand() - 0.5))/(2*np.pi)

    def reset(self):
        for spin in self.spin_list: spin.reset()

    def set_m(self,m):
        for spin in self.spin_list: spin.m = m

    # Methods the same as SpinGroup:
    def scale_m_signal(self, scale):
        for spin in self.spin_list: spin.scale_m_signal(scale)

    def get_m(self):
        m = np.array([[0.0], [0.0], [0.0]])
        for spin in self.spin_list:
            m += spin.m
        m /= self.num_spins
        return m

    def apply_ideal_RF(self, rf_phase, fa, f_low, f_high, gradients):
        for spin in self.spin_list: spin.apply_ideal_RF(rf_phase,fa,f_low,f_high,gradients)

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
        avg_signal = np.zeros(np.array(self.spin_list[0].signal).shape, dtype=complex)
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
        magnetizations_total = np.zeros((3, len(pulse_shape)))
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


###

if __name__ == "__main__":
    # Generate a spin echo. Compare spins with / without T2star modeling
    TE = 200e-3 # 200
    system = Opts()
    rf90 = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=2e-3,
                           slice_thickness=5e-3, apodization=0.5, time_bw_product=4)
    rf180 = make_sinc_pulse(flip_angle=np.pi, system=system, duration=2e-3,
                           slice_thickness=5e-3, apodization=0.5, time_bw_product=4)

    delay1 = make_delay((TE/2) - 0.5*calc_duration(rf90) - 0.5*calc_duration(rf180))
    delay2 = make_delay((TE/2) - 0.5*calc_duration(rf180))
    delay3 = make_delay(TE - 0.5*calc_duration(rf90))

    # Spin echo config.
    seq_SE = Sequence()
    seq_SE.add_block(rf90)
    seq_SE.add_block(delay1)
    seq_SE.add_block(rf180)
    seq_SE.add_block(delay2)

    # Gradient echo config.
    seq_GRE = Sequence()
    seq_GRE.add_block(rf90)
    seq_GRE.add_block(delay3)



    # Store seq info
    seq_info_SE = blcsim.store_pulseq_commands(seq_SE)
    seq_info_GRE = blcsim.store_pulseq_commands(seq_GRE)

    # Simulate
    # First, SE
    # sanity test this new group
    sg0 = SpinGroupT2star(loc=(0,0,0), pdt1t2=(1,2,0.2), t2star=0, num_spins=20)
    sg1 = SpinGroupT2star(loc=(0,0,0), pdt1t2=(1,2,0.2), t2star=0.1, num_spins=20)

    blcsim.apply_pulseq_commands(sg0, seq_info_SE, store_m=False)
    signal0_SE = sg0.get_m_signal()
    blcsim.apply_pulseq_commands(sg1, seq_info_SE, store_m=False)
    signal1_SE = sg1.get_m_signal()



    # Second, GRE
    # sanity test this new group
    sg0 = SpinGroupT2star(loc=(0,0,0), pdt1t2=(1,2,0.2), t2star=0, num_spins=20)
    sg1 = SpinGroupT2star(loc=(0,0,0), pdt1t2=(1,2,0.2), t2star=0.1, num_spins=20)

    blcsim.apply_pulseq_commands(sg0, seq_info_GRE, store_m=False)
    signal0_GRE = sg0.get_m_signal()
    blcsim.apply_pulseq_commands(sg1, seq_info_GRE, store_m=False)
    signal1_GRE = sg1.get_m_signal()

    savemat('t2star_sim_test.mat',{'s0gre':signal0_GRE, 's1gre':signal1_GRE,
                                   's0se':signal0_SE, 's1se':signal1_SE})