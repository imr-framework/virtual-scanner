#!/usr/bin/env python3
#
# Compatibility layer vaguely similar to a certain other company's
# pulse programming language
#

import numpy as np
import experiment as exp

class Marcostek:
    """Provides a simple API to interact with the Experiment class, with
    discrete commands that can be called in order."""

    def __init__(self, exp, grad_update_interval=5,
                 tx_gate_overhead=0,
                 invert_tx_gate=False,
                 rx_gate_overhead=0,
                 invert_rx_gate=False):
        """exp: previously-created Experiment object

        grad_update_interval: raster time of gradients in us.

        Must be equal to or greater than 1 / grad_max_update_rate used
        to create the exp object (by default 0.2 MSPS, so
        grad_update_interval must be > 5 for OCRA1 and > 1.25 for
        GPA-FHDO)

        tx_gate_overhead: how far in advance of an RF pulse to turn the TX gate TTL on

        invert_tx_gate: invert TX gate TTL polarity (if true, logic 0 during TX)

        rx_gate_overhead: how far in advance of an acquisition (RX) pulse to turn the RX gate TTL on

        invert_rx_gate: invert RX gate TTL polarity (if true, logic 0 during RX)
        """
        self._exp = exp
        self._grad_update_interval = grad_update_interval

        assert tx_gate_overhead >= 0, "TX gate overhead time cannot be negative"
        self._tx_gate_overhead = tx_gate_overhead
        self._invert_tx_gate = invert_tx_gate

        assert rx_gate_overhead >= 0, "RX gate overhead time cannot be negative"
        self._rx_gate_overhead = rx_gate_overhead
        self._invert_rx_gate = invert_rx_gate
        self._global_time = grad_update_interval


    def _chan_str(self, chan):
        chan_str = ['x', 'y', 'z', 'z2']
        if type(chan) is str:
            assert chan in chan_str, "Unknown grad channel"
            return chan
        else:
            assert 0 <= chan and chan <= 3, "Unknown grad channel index"
            return chan_str[chan]

    ### Pulse programming commands

    def delay(self, time):
        """ time in us """
        assert time > 0, "Time cannot be negative"
        self._global_time += time

    def gradoff(self, chan):
        """sets a gradient channel to 0V for the ocra1, 0A for the gpa-fhdo.
        chan: 0, 1, 2 or 3, or 'x', 'y', 'z' or 'z2'

        Takes grad_update_interval us to complete.
        """
        self._exp.add_flodict({'grad_v' + self._chan_str(chan):
                               ( np.array([self._global_time]), np.array([0]) )})

        self._global_time += self._grad_update_interval

    def gradon(self, chan, value):
        """sets a gradient channel to value
        chan: 0, 1, 2 or 3, or 'x', 'y', 'z' or 'z2'
        value: between -1 and +1; +1 = maximum gradient output, -1 = maximum negative gradient output

        Takes grad_update_interval us to complete.
        """
        assert -1 <= value and value <= 1, "Grad value out of range"
        self._exp.add_flodict({'grad_v' + self._chan_str(chan):
                               ( np.array([self._global_time]), np.array([value]) )})

        self._global_time += self._grad_update_interval

    def gradramp(self, chan, start_val, end_val, n_steps, step_duration):
        """Creates a ramp on a gradient channel
        chan: 0, 1, 2 or 3, or 'x', 'y', 'z' or 'z2'

        start_val and end_val: between -1 and +1; +1 = maximum
        gradient output, -1 = maximum negative gradient output

        n_steps: number of output steps

        step_duration: time duration of each step, at least grad_update_interval

        Outputs will occur at times step_duration, 2*step_duration,
        etc with values of start_val + (end_val - start_val) * k /
        (n_steps - 1), where k goes from 0 to n_steps - 1

        Note that the total ramp length is step_duration * n_steps
        """
        assert -1 <= start_val and start_val <= 1, "Grad ramp start value out of range"
        assert -1 <= end_val and end_val <= 1, "Grad ramp end value out of range"
        assert n_steps > 0, "Number of steps cannot be negative"
        assert step_duration >= self._grad_update_interval, \
            "Step duration cannot be shorter than gradient update interval"
        self._exp.add_flodict({'grad_v' + self._chan_str(chan):
                               ( self._global_time + np.linspace(step_duration, n_steps * step_duration, n_steps),
                                 np.linspace(start_val, end_val, n_steps) )})
        self._global_time += n_steps * step_duration

    def pulse(self, chan, amp, phase, duration, end_amp=0, end_phase=0, pulse_tx_gate=True, tx_gate_overhead=None):
        """Does a rectangular RF pulse

        chan: 0 or 1, marga TX channel to output the pulse on

        amp: output amplitude, proportional to voltage; 1 = full-scale

        phase: RF phase (w.r.t. carrier tone) in degrees; wrapped to [0, 360) if out of range

        duration: length of RF pulse

        end_amp: final output amplitude to return to, if a nonzero
        value is desired (can be used to create pulses that take place
        while other events are occurring)

        end_phase: phase used for end_amp in degrees; wrapped to [0, 360) if out of range

        pulse_tx_gate: pulse the TX gate TTL output during the RF pulse

        tx_gate_overhead: time before RF pulse begins to turn on the TX gate TTL

        Note: total delay of the command is duration + tx_gate_overhead

        Note: if pulse_tx_gate is true and two pulses are run one
        after the other, the TX gate value is undefined at the instant
        between pulses (one is trying to switch it off, while the
        other is trying to switch it on). To avoid this happening,
        either add a delay between pulses, or make sure that one of
        the pulses' pulse_tx_gate parameter is set to False.
        """
        if tx_gate_overhead is None:
            tx_gate_overhead = self._tx_gate_overhead
        assert chan in [0, 1], "Invalid RF channel"
        assert 0 <= amp and amp <= 1, "RF amplitude out of range"
        assert 0 <= end_amp and amp <= 1, "RF end amplitude out of range"
        assert duration > 0, "RF pulse duration cannot be negative"
        assert tx_gate_overhead >= 0, "RF pulse gate overhead time cannot be negative"

        phase_rad, end_phase_rad = phase * np.pi/180, end_phase * np.pi/180
        cplx_scale = np.cos(phase_rad) + 1j * np.sin(phase_rad)
        end_cplx_scale = np.cos(end_phase_rad) + 1j * np.sin(end_phase_rad)

        self._exp.add_flodict({'tx'+str(chan): ( self._global_time + tx_gate_overhead + np.array([0, duration]),
                                                 np.array([amp * cplx_scale, end_amp * end_cplx_scale]) )})

        if pulse_tx_gate:
            self._exp.add_flodict({'tx_gate': ( self._global_time + np.array([0, duration + tx_gate_overhead]),
                                                np.array([not self._invert_tx_gate, self._invert_tx_gate]) )})

        self._global_time += duration + tx_gate_overhead

    def rx(self, chan, duration, pulse_rx_gate=True, rx_gate_overhead=None):
        """Acquire RX data for the duration specified, at the RX rate
        specified when the Experiment object was created.

        chan: 0 or 1, marga RX channel to sample

        duration: length of acquisition

        pulse_rx_gate: pulse the RX gate TTL output during the acquisition

        rx_gate_overhead: time before acquisition begins to turn on the RX gate TTL

        Note: total delay of the command is duration + rx_gate_overhead
        """

        if rx_gate_overhead is None:
            rx_gate_overhead = self._rx_gate_overhead
        assert chan in [0, 1], "Invalid RF channel"
        assert duration > 0, "RX duration cannot be negative"
        assert rx_gate_overhead >= 0, "RX gate overhead time cannot be negative"

        self._exp.add_flodict({
            'rx'+str(chan)+'_en': (self._global_time + rx_gate_overhead + np.array([0, duration]),
                                   np.array([1, 0]) )})

        if pulse_rx_gate:
            self._exp.add_flodict({'rx_gate': (self._global_time + np.array([0, duration + rx_gate_overhead]),
                                               np.array([not self._invert_rx_gate, self._invert_rx_gate]) ) })

        self._global_time += duration + rx_gate_overhead

def test_marcostek():
    expt = exp.Experiment(lo_freq=5, # MHz
                          rx_t=1.5) # us sampling rate)
    f = Marcostek(expt)

    # Turn all 4 gradients off
    for k in range(2):
        f.gradoff(k) # index-based
    f.gradoff('z') # string-based
    f.gradoff('z2')

    # Wait 10us
    f.delay(10)

    # Turn gradients on, 5us apart, to -1, -0.5, 0.3 and 0.8x full-scale
    for k, m in enumerate([-1, -0.5, 0.3, 0.8]):
        f.gradon(k, m)

    # Ramp the x and y gradients one after the other: start value, end value, number of steps, step duration
    f.delay(20)
    f.gradramp('x', 0.0, 0.8, 5, 8)
    f.gradramp('y', 0.0, 0.6, 5, 8)

    f.delay(20)
    f.gradramp('x', 0.8, -0.5, 10, 5)
    f.gradramp('y', 0.6, -0.5, 10, 5)

    # Do some RF pulses of different amplitudes and lengths: 0, 180 and 235 deg phases
    # Channel, amplitude, phase, pulse length
    first_time = 50
    second_time = 30
    third_time = 70
    f.delay(20)
    f.pulse(0, 0.5, 0, first_time)
    f.delay(10)
    f.pulse(1, 0.3, 0, first_time)

    f.delay(10)
    f.pulse(0, 0.9, 180, second_time, pulse_tx_gate=False)
    f.delay(20)
    f.pulse(1, 0.8, 180, second_time)

    f.delay(30)
    f.pulse(0, 0.4, 235, third_time)
    f.delay(15)
    f.pulse(1, 0.6, 235, third_time)

    # Acquire data on channel 0 for 100us (at 1.5us sampling rate, specified in the construction of expt)
    f.rx(0, 100)

    # Turn off gradients
    for k in range(4):
        f.gradoff(k)

    rxd, msgs = expt.run()
    expt.close_server(True)
    expt._s.close() # close socket

if __name__ == "__main__":
    test_marcostek()
