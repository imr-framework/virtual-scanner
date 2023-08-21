#!/usr/bin/env python3
# Basic script to plot the CSV that marga_sim produces, so that you can visualise the expected pulse sequence from the hardware.

import numpy as np
import matplotlib.pyplot as plt
import sys, pdb
from local_config import fpga_clk_freq_MHz
st = pdb.set_trace

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python {:s} <csv_file.csv>".format(sys.argv[0]) )
        print("\t Example: python {:s} /tmp/marga_sim.csv".format(sys.argv[0]) )
        exit()

    # data = np.genfromtxt(sys.argv[1],
    #                      dtype=None,
    #                      comments='#',
    #                      delimiter=',',
    #                      skip_header=0,
    #                      usecols=(0,1),
    #                      names=True)

    data = np.loadtxt(sys.argv[1], skiprows=2, delimiter=',')
    data[1:, 0] = data[1:, 0] - data[1, 0] + 1 # remove dead time in the beginning taken up by simulated memory writes

    time_us = data[:,0]/fpga_clk_freq_MHz
    tx = data[:,1:5].astype(np.int16) / 32768
    # offset binary
    fhdo = data[:,5:9].astype(np.uint16) / 32768 - 1
    # 2's complement
    ocra1 = ( (data[:,9:13].astype(np.int32) ^ (1 << 17)) - (1 << 17) ).astype(np.int32) / 131072
    gdata = np.hstack([fhdo, ocra1])
    gdata_nonzero = np.nonzero(gdata.any(0))[0]

    rx = data[:,14:21].astype(np.uint8)
    rx_en = rx[:, 5:] # ignore the rate logic, only plot the RX enables
    io = data[:,21:].astype(float)
    io[:,-1] = io[:,-1] / 256 # scale down LEDs to [0, 1)

    fig, axes = plt.subplots(4, 1, figsize=(12,8), sharex='col')

    (txs, grads, rxs, ios) = axes

    txs.step(time_us, tx, where='post')
    txs.legend(['tx0 i', 'tx0 q', 'tx1 i', 'tx1 q'])

    glegends = ['fhdo x', 'fhdo y', 'fhdo z', 'fhdo z2', 'ocra1 x', 'ocra1 y', 'ocra1 z', 'ocra1 z2']

    if gdata_nonzero.size != 0:
        grads.step(time_us, gdata[:, gdata_nonzero], where='post')
        grads.legend([glegends[k] for k in gdata_nonzero])

    rxs.step(time_us, rx_en, where='post')
    rxs.legend(["rx0 en", "rx1 en"])

    ios.step(time_us, io, where='post')
    ios.legend(['tx gate', 'rx gate', 'trig out', 'leds'])
    ios.set_xlabel('time (us)')

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    plt.show()
