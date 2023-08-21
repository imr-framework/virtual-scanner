#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import experiment as ex
from local_config import grad_board

import pdb
st = pdb.set_trace

def test_jitter(
        rf_pulse_freq = 2, # MHz
        rf_pulse_start_time = 10, # us
        rf_pulse_length = 2, # us
        rf_amp=0.8, # fraction of full-scale
        rf_interval=5, # us, delay between start of first and start of
                       # second pulse - must be bigger than pulse
                       # length
        grad_pulse_start_time = 10, # us
        grad_pulse_length = 40, # us
        grad_interval=80, # us
        grad_amp=0.2, # fraction of full-scale
        loops=10,
        print_loops=False,
        plot_sequence=False,
        experiment_kwargs={}):
    """Two successive pulses, with an adjustable interval between them.

    Trigger an oscilloscope from the TX gate or first pulse, and look
    at the second pulse, with a suitable hold-off time so that
    triggering is reliable. Over many repetitions, the timing jitter
    will be visible.
    """
    tx_gate_start_time = 0
    tx_gate_length = 0.1

    rf_data = (
        rf_pulse_start_time + np.array([0, rf_pulse_length, rf_interval, rf_pulse_length + rf_interval]),
        np.array([rf_amp, 0, rf_amp, 0])
    )

    grad_data = (
        grad_pulse_start_time + np.array([0, grad_pulse_length, grad_interval, grad_pulse_length + grad_interval]),
        np.array([grad_amp, 0, grad_amp, 0])
    )

    gpa_fhdo_offset_time = 10 if grad_board == "gpa-fhdo" else 0

    event_dict = {'tx0': rf_data,
                  'tx1': rf_data,
                  'grad_vx': grad_data,
                  'grad_vy': grad_data,
                  'grad_vz': grad_data,
                  'grad_vz2': grad_data,
                  'tx_gate': (tx_gate_start_time + np.array([0, tx_gate_length]), np.array([1, 0]))}

    exp = ex.Experiment(lo_freq=rf_pulse_freq, gpa_fhdo_offset_time=gpa_fhdo_offset_time, **experiment_kwargs)
    exp.add_flodict(event_dict)

    if plot_sequence:
        exp.plot_sequence()
        plt.show()

    for k in range(loops):
        exp.run()
        if print_loops and k % 1000 == 0:
            print(f"Loop {k}")

    exp.close_server(only_if_sim=True)

if __name__ == "__main__":
    # slowed down to suit VN's setup, due to GPA-FHDO communication chip being dead
    experiment_kwargs = {'init_gpa': True, 'grad_max_update_rate': 0.1}
    loops = int(1e6)

    # Enable the test that you want to run below

    # RF pulses close together, gradient pulses further apart, rf and grad start simultaneously (default)
    if False:
        test_jitter(loops=loops, experiment_kwargs=experiment_kwargs) # many loops

    # RF and gradient pulses start almost simultaneously (RF offset to avoid SPI noise), second pulses happen at an adjustable interval
    if True:
        interval=100000 # 100ms
        amp = 0.2
        test_jitter(loops=loops, rf_pulse_start_time=10.5,
                    rf_interval=interval, rf_amp=amp,
                    grad_interval=interval, grad_amp=amp,
                    experiment_kwargs=experiment_kwargs) # many loops
