#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import console.server.scanner_control.seq.marcos_client.experiment as ex
from console.server.scanner_control.seq.marcos_client.local_config import grad_board

import pdb
st = pdb.set_trace

def trapezoid(plateau_a, total_t, ramp_t, ramp_pts, total_t_end_to_end=True, base_a=0):
    """Helper function that just generates a Numpy array starting at time
    0 and ramping down at time total_t, containing a trapezoid going from a
    level base_a to plateau_a, with a rising ramp of duration ramp_t and
    sampling period ramp_ts."""

    # ramp_pts = int( np.ceil(ramp_t/ramp_ts) ) + 1
    rise_ramp_times = np.linspace(0, ramp_t, ramp_pts)
    rise_ramp = np.linspace(base_a, plateau_a, ramp_pts)

    # [1: ] because the first element of descent will be repeated
    descent_t = total_t - ramp_t if total_t_end_to_end else total_t
    t = np.hstack([rise_ramp_times, rise_ramp_times[:-1] + descent_t])
    a = np.hstack([rise_ramp, np.flip(rise_ramp)[1:]])
    return t, a

def trap_cent(centre_t, plateau_a, trap_t, ramp_t, ramp_pts, base_a=0):
    """Like trapezoid, except it generates a trapezoid shape around a centre
    time, with a well-defined area given by its amplitude (plateau_a)
    times its time (trap_t), which is defined from the start of the
    ramp-up to the start of the ramp-down, or (equivalently) from the
    centre of the ramp-up to the centre of the ramp-down. All other
    parameters are as for trapezoid()."""
    t, a = trapezoid(plateau_a, trap_t, ramp_t, ramp_pts, False, base_a)
    return t + centre_t - (trap_t + ramp_t)/2, a

def grad_echo(trs=21, plot_rx=False, init_gpa=False, plot_sequence=False,
              dbg_sc=0.5, # set to 0 to avoid 2nd RF debugging pulse, otherwise amp between 0 or 1
              lo_freq=0.1, # MHz
              rf_amp=1, # 1 = full-scale

              slice_amp=0.4, # 1 = gradient full-scale
              phase_amp=0.3, # 1 = gradient full-scale
              readout_amp=0.8, # 1 = gradient full-scale
              rf_duration=50,
              trap_ramp_duration=50, # us, ramp-up/down time
              trap_ramp_pts=5, # how many points to subdivide ramp into
              phase_delay=100, # how long after RF end before starting phase ramp-up
              phase_duration=200, # length of phase plateau
              tr_wait=100, # delay after end of RX before start of next TR

              rx_period=10/3 # us, 3.333us, 300 kHz rate
              ):
    ## All times are in the context of a single TR, starting at time 0

    phase_amps = np.linspace(phase_amp, -phase_amp, trs)

    rf_tstart = 100 # us
    rf_tend = rf_tstart + rf_duration # us

    slice_tstart = rf_tstart - trap_ramp_duration
    slice_duration = (rf_tend - rf_tstart) + 2*trap_ramp_duration # includes rise, plateau and fall
    phase_tstart = rf_tend + phase_delay
    readout_tstart = phase_tstart
    readout_duration = phase_duration*2

    rx_tstart = readout_tstart + trap_ramp_duration # us
    rx_tend = readout_tstart + readout_duration - trap_ramp_duration # us

    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends

    tr_total_time = readout_tstart + readout_duration + tr_wait + 7000 # start-finish TR time

    def grad_echo_tr(tstart, pamp):
        gvxt, gvxa = trapezoid(slice_amp, slice_duration, trap_ramp_duration, trap_ramp_pts)
        gvyt, gvya = trapezoid(pamp, phase_duration, trap_ramp_duration, trap_ramp_pts)

        gvzt1 = trapezoid(readout_amp, readout_duration/2, trap_ramp_duration, trap_ramp_pts)
        gvzt2 = trapezoid(-readout_amp, readout_duration/2, trap_ramp_duration, trap_ramp_pts)
        gvzt = np.hstack([gvzt1[0], gvzt2[0] + readout_duration/2])
        gvza = np.hstack([gvzt1[1], gvzt2[1]])

        rx_tcentre = (rx_tstart + rx_tend) / 2

        value_dict = {
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,   rx_tcentre - 10, rx_tcentre + 10]) + tstart,
                     np.array([rf_amp,0,  dbg_sc*(1 + 0.5j),0]) ),

            'tx1': ( np.array([rx_tstart + 15, rx_tend - 15]) + tstart, np.array([dbg_sc * pamp * (1 + 0.5j), 0]) ),
            'grad_vx': ( gvxt + tstart + slice_tstart, gvxa ),
            'grad_vy': ( gvyt + tstart + phase_tstart, gvya),
            'grad_vz': ( gvzt + tstart + readout_tstart, gvza),
            'rx0_en': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
            'rx1_en': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ), # acquire on RX1 for example too
            'rx_gate': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
            'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin)

    tr_t = 0 # start the first TR at 20us
    for pamp in phase_amps:
        expt.add_flodict( grad_echo_tr( tr_t, pamp) )
        tr_t += tr_total_time

    if plot_sequence:
        expt.plot_sequence()
        plt.show()

    rxd, msgs = expt.run()
    expt.close_server(True)
    expt._s.close() # close socket

    if plot_rx:
        plt.plot( rxd['rx0'].real )
        plt.plot( rxd['rx0'].imag )
        plt.plot( rxd['rx1'].real )
        plt.plot( rxd['rx1'].imag )
        plt.show()

def turbo_spin_echo(plot_rx=False, init_gpa=False, plot_sequence=False,
                    dbg_sc=0.5, # set to 0 to avoid RF debugging pulses in each RX window, otherwise amp between 0 or 1
                    lo_freq=0.2, # MHz
                    rf_amp=1, # 1 = full-scale

                    rf_pi2_duration=50, # us, rf pi/2 pulse length
                    rf_pi_duration=None, # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration

                    # trapezoid properties - shared between all gradients for now
                    trap_ramp_duration=50, # us, ramp-up/down time
                    trap_ramp_pts=5, # how many points to subdivide ramp into

                    # spin-echo properties
                    echos_per_tr=5, # number of spin echoes (180 pulses followed by readouts) to do
                    echo_duration=2000, # us, time from the centre of one echo to centre of the next

                    readout_amp=0.8, # 1 = gradient full-scale
                    readout_duration=500, # us, time in the centre of an echo when the readout occurs
                    rx_period=10/3, # us, 3.333us, 300 kHz rate
                    readout_grad_duration=700, # us, readout trapezoid lengths (mid-ramp-up to mid-ramp-down)
                    # (must at least be longer than readout_duration + trap_ramp_duration)

                    phase_start_amp=0.6, # 1 = gradient full-scale, starting amplitude (by default ramps from +ve to -ve in each echo)
                    phase_grad_duration=150, # us, phase trapezoid lengths (mid-ramp-up to mid-ramp-down)
                    phase_grad_interval=1200, # us, interval between first phase trapezoid and its negative-sign counterpart within a single echo

                    # slice trapezoid timing is the same as phase timing
                    slice_start_amp=0.3, # 1 = gradient full-scale, starting amplitude (by default ramps from +ve to -ve in each TR)

                    tr_pause_duration=3000, # us, length of time to pause from the end of final echo's RX pulse to start of next TR
                    trs=5 # number of TRs
                    ):
    """
    readout gradient: x
    phase gradient: y
    slice/partition gradient: z
    """

    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration

    phase_amps = np.linspace(phase_start_amp, -phase_start_amp, echos_per_tr)
    slice_amps = np.linspace(slice_start_amp, -slice_start_amp, trs)

    # create appropriate waveforms for each echo, based on start time, echo index and TR index
    # note: echo index is 0 for the first interval (90 pulse until first 180 pulse) thereafter 1, 2 etc between each 180 pulse
    def rf_wf(tstart, echo_idx):
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                             tstart + echo_duration - rf_pi_duration/2]), np.array([pi2_phase, 0, pi_phase]) * rf_amp
        elif echo_idx == echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2, tstart + echo_duration - rf_pi_duration/2]), np.array([0, pi_phase]) * rf_amp

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 2 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2 - tx_gate_pre,
                             tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                             tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                             np.array([1, 0, 1])
        elif echo_idx == echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post, tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                np.array([0, 1])

    def readout_grad_wf(tstart, echo_idx):
        if echo_idx == 0:
            return trap_cent(tstart + echo_duration*3/4, readout_amp, readout_grad_duration/2,
                             trap_ramp_duration, trap_ramp_pts)
        else:
            return trap_cent(tstart + echo_duration/2, readout_amp, readout_grad_duration,
                             trap_ramp_duration, trap_ramp_pts)

    def readout_wf(tstart, echo_idx):
        if echo_idx != 0:
            return np.array([tstart + (echo_duration - readout_duration)/2, tstart + (echo_duration + readout_duration)/2 ]), np.array([1, 0])
        else:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise

    def phase_grad_wf(tstart, echo_idx):
        t1, a1 = trap_cent(tstart + (echo_duration - phase_grad_interval)/2, phase_amps[echo_idx-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (echo_duration + phase_grad_interval)/2, -phase_amps[echo_idx-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])

    def slice_grad_wf(tstart, echo_idx, tr_idx):
        t1, a1 = trap_cent(tstart + (echo_duration - phase_grad_interval)/2, slice_amps[tr_idx], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (echo_duration + phase_grad_interval)/2, -slice_amps[tr_idx], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])

    tr_total_time = echo_duration * (echos_per_tr + 1) + tr_pause_duration

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin))

    global_t = 0 # start the first TR at this time

    for tr in range(trs):
        for echo in range(echos_per_tr + 1):
            tx_t, tx_a = rf_wf(global_t, echo)
            tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo)
            readout_t, readout_a = readout_wf(global_t, echo)
            rx_gate_t, rx_gate_a = readout_wf(global_t, echo)

            readout_grad_t, readout_grad_a = readout_grad_wf(global_t, echo)
            phase_grad_t, phase_grad_a = phase_grad_wf(global_t, echo)
            slice_grad_t, slice_grad_a = slice_grad_wf(global_t, echo, tr)

            global_t += echo_duration

            expt.add_flodict({
                'tx0': (tx_t, tx_a),
                'tx1': (tx_t, tx_a),
                'grad_vx': (readout_grad_t, readout_grad_a),
                'grad_vy': (phase_grad_t, phase_grad_a),
                'grad_vz': (slice_grad_t, slice_grad_a),
                'rx0_en': (readout_t, readout_a),
                'rx1_en': (readout_t, readout_a),
                'tx_gate': (tx_gate_t, tx_gate_a),
                'rx_gate': (rx_gate_t, rx_gate_a),
            })

        global_t += tr_pause_duration

    if plot_sequence:
        expt.plot_sequence()
        plt.show()

    rxd, msgs = expt.run()
    expt.close_server(True)
    expt._s.close() # close socket

    if plot_rx:
        plt.plot( rxd['rx0'].real )
        plt.plot( rxd['rx0'].imag )
        plt.plot( rxd['rx1'].real )
        plt.plot( rxd['rx1'].imag )
        plt.show()

def radial(trs=36, plot_rx=False, init_gpa=False, plot_sequence=False):
    ## All times are relative to a single TR, starting at time 0
    lo_freq = 0.2 # MHz
    rf_amp = 0.5 # 1 = full-scale

    G = 0.5 # Gx = G cos (t), Gy = G sin (t)
    angles = np.linspace(0, 2*np.pi, trs) # angle

    gradz_tstart = 0 # us
    grady_tstart = gradz_tstart # us
    rf_tstart = 5 # us
    rf_tend = 50 # us
    rx_tstart = 70 # us
    rx_tend = 180 # us
    rx_period = 3 # us
    tr_total_time = 220 # start-finish TR time
    # tr_total_time = 5000 # start-finish TR time

    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends

    def radial_tr(tstart, th):
        dbg_sc=0.01
        gx = G * np.cos(th)
        gy = G * np.sin(th)

        value_dict = {
            # second tx0 pulse and tx1 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,    rx_tstart + 15, rx_tend - 15]),
                     np.array([rf_amp, 0,    dbg_sc * (gx+gy*1j), 0]) ),
            'tx1': ( np.array([rx_tstart + 15, rx_tend - 15]), np.array([dbg_sc * (gx+gy*1j), 0]) ),
            'grad_vz': ( np.array([gradz_tstart]),
                         np.array([gx]) ),
            'grad_vy': ( np.array([grady_tstart]),
                         np.array([gy]) ),
            'rx0_en' : ( np.array([rx_tstart, rx_tend]),
                            np.array([1, 0]) ),
            'rx_gate' : ( np.array([rx_tstart, rx_tend]),
                            np.array([1, 0]) ),
            'tx_gate' : ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]),
                          np.array([1, 0]) )
            }

        for k, v in value_dict.items():
            # v for read, value_dict[k] for write
            value_dict[k] = (v[0] + tstart, v[1])

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin))

    tr_t = 0 # start the first TR at 0us
    for th in angles:
        expt.add_flodict( radial_tr( tr_t, th ) )
        tr_t += tr_total_time

    if plot_sequence:
        expt.plot_sequence()
        plt.show()

    rxd, msgs = expt.run()
    expt.close_server(True)
    expt._s.close() # close socket

    if plot_rx:
        plt.plot( rxd['rx0'].real )
        plt.plot( rxd['rx0'].imag )
        # plt.plot( rxd['rx1'].real )
        # plt.plot( rxd['rx1'].imag )
        plt.show()

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('test_grad_echo_loop()')
    # for k in range(100):
    # grad_echo(lo_freq=1, trs=1, plot_rx=True, init_gpa=True, dbg_sc=1)
    radial(trs=100, init_gpa=True, plot_rx=True)
    # turbo_spin_echo(trs=2, echos_per_tr=4, plot_sequence=False, tr_pause_duration=100000)

    if False:
        # Stress test: run lots of sequences on the server - should
        # take around a day
        k = 0
        while k < 100000:
            k += 1
            print("TSE, trial {:d}".format(k))
            turbo_spin_echo(lo_freq=0.2,
                            trs=2, echos_per_tr=6,
                            rf_amp=0.7,
                            echo_duration=1000,
                            readout_amp=0.2,
                            readout_duration=300,
                            readout_grad_duration=450,
                            phase_grad_duration=150,
                            phase_grad_interval=600,
                            tr_pause_duration=2000,
                            init_gpa=True, plot_rx=False)
