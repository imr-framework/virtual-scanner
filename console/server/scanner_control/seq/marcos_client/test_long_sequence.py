#!/usr/bin/env python3
#
# Test for verifying whether once a long, complex sequence has been started, it can run to completion and/or be successfully aborted
#
# Test procedure for abortion:
# - start marcos_server on hardware
# - run this file - it just slowly increments the LEDs and does a loopback for a long time
# - halt marcos_server explicitly on hardware; this file will crash due to the server going down
# - restart marcos_server
# - run this file again, and verify that the LED pattern starts again from 0
#
# If you wish to see what happens for different delays, just alter the arguments to long_loopback() at the end of this file.

import numpy as np
import matplotlib.pyplot as plt
import experiment as ex
from local_config import grad_board

import pdb
st = pdb.set_trace

# In each TR, a wait of rf_start_delay occurs, the RF turns on, it
# turns off after rf_length. Each TR is by default 1 second long. The RX
# turns on rx_pad microseconds before and turns off rx_pad
# microseconds after the RF pulse.

lo_freq = 5 # MHz
rx_period = 5 # us
rx_pad = 20 # us
rf_start_delay = 100 # us
rf_amp = 0.4
rf_length = 200 # us

def long_loopback(rf_interval=1000000, trs=20):

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, halt_and_reset=True)

    for k in range(trs):
        rf_t = rf_start_delay + k*rf_interval + np.array([0, rf_length])
        rx_t = rf_start_delay + k*rf_interval - rx_pad + np.array([0, rf_length + 2*rx_pad])
        expt.add_flodict({
            'tx1': ( rf_t, np.array([rf_amp, 0]) ),
            'rx1_en': ( rx_t, np.array([1, 0]) )
            # 'leds': ( np.array([k*rf_interval]), np.array(k) )
            })

    rxd, msgs = expt.run()
    expt.close_server(only_if_sim=True)

    expt._s.close() # close socket on client

if __name__ == "__main__":
    long_loopback(1000000, 255) #
