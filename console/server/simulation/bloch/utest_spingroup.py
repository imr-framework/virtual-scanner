# Proper unittest unit tests

import unittest

import virtualscanner.server.simulation.bloch.spingroup_ps as sg

import pypulseq.Sequence.sequence
from pypulseq.opts import Opts

from pypulseq.make_sinc import make_sinc_pulse
import numpy as np
from math import pi
# Unit tests

GAMMA_BAR = 42.58e6

class TestSpinGroupFunctions(unittest.TestCase):

    def test_gradient(self):
        spin = sg.SpinGroup(loc=(1,0,0),pdt1t2=(1,0,0))
        spin.m =  np.array([[1],[0],[0]])
        spin.fpwg(grad_area=np.array([5,0,0]), t=0.01)
        self.assertEqual(spin.get_m_signal(), np.exp(-1j*(GAMMA_BAR*2*pi)*5))


    def test_rf(self):
        # Test a 90 degree sinc pulse
        spin = sg.SpinGroup(loc=(0,0,0),pdt1t2=(1,0,0))
        # Make pulseq rf shape
        kwargs_for_opts = {"rf_ring_down_time": 0, "rf_dead_time": 0}
        system = Opts(kwargs_for_opts)
        thk = 5e-3
        flip = 90 * pi / 180
        kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 4e-3, "slice_thickness": thk,
                           "apodization": 0.5, "time_bw_product": 4}
        rf, g_ss = make_sinc_pulse(kwargs_for_sinc, 2)
        g = g_ss.amplitude / GAMMA_BAR
        spin.apply_rf(pulse_shape = np.array(rf.signal[0]/GAMMA_BAR),
                      grads_shape = np.tile([[0],[0],[g]],len(rf.signal[0])),
                      dt = system.rf_raster_time)

        np.testing.assert_allclose(spin.get_m_signal(), 1j, rtol=1e-3)


    def test_adc(self):
        # Test for uniform phase difference during readout under constant gradient

        spin = sg.SpinGroup(loc=(1,0,0),pdt1t2=(1,0,0))
        spin.m = np.array([[1],[0],[0]])
        dt = 10e-6
        spin.readout(dwell=dt, n=10, delay=0,
                     grad=np.tile([[0.02],[0],[0]], 11), timing=np.arange(0,11*dt,dt))


        np.testing.assert_allclose(np.squeeze(spin.signal),
                                      np.exp(-1j*2*pi*GAMMA_BAR*0.02*np.arange(dt,11*dt,dt)))
        # make commands
        # apply commands to spin group


    def test_relaxation(self):
        # T1 = 1000 ms, T2 = 100 ms
        spin = sg.SpinGroup(loc=(0,0,0),pdt1t2=(1,1,0.1))
        spin.m = np.array([[1],[0],[0]])
        spin.fpwg(grad_area=np.array([0,0,0]),t=0.5)

        np.testing.assert_array_equal(spin.m, [[np.exp(-5)],[0],[1-np.exp(-0.5)]])


if __name__ == "__main__":
    unittest.main()
