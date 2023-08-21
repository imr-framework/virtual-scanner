#!/usr/bin/env python3

from os import device_encoding
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import time

import sys
import console.server.scanner_control.seq.adjustments_acq.config as cfg  # pylint: disable=import-error
import console.server.scanner_control.seq.marcos_client.experiment as ex  # pylint: disable=import-error
from console.server.scanner_control.seq.marcos_client.examples import trap_cent  # pylint: disable=import-error
import console.server.scanner_control.seq.adjustments_acq.scripts as scr  # pylint: disable=import-error
from console.utils import constants

def larmor_step_search(step_search_center=cfg.LARMOR_FREQ, steps=30, step_bw_MHz=5e-3, plot=False,
                       shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z, delay_s=1, gui_test=False):
    """
    Run a stepped search through a range of frequencies to find the highest signal response
    Used to find a starting point, not for precision

    Args:
        step_search_center (float): [MHz] Center for search, defaults to config LARMOR_FREQ
        steps (int): Number of search steps
        step_bw_MHz (float): [MHz] Distance in MHz between each step
        plot (bool): Default False, plot final data
        shim_x, shim_y, shim_z (float): Shim value, defaults to config SHIM_ values, must be less than 1 magnitude
        delay_s (float): Delay between readings in seconds
        gui_test (bool): Default False, takes dummy data instead of actual data for GUI testing away from scanner

    Returns:
        float: Estimated larmor frequency in MHz
        dict: Dictionary of data
    """

    # Pick out the frequencies to run through
    swept_freqs = np.linspace(step_search_center - ((steps - 1) / 2 * step_bw_MHz),
                              step_search_center + ((steps - 1) / 2 * step_bw_MHz), num=steps)
    larmor_freq = swept_freqs[0]
    # Set the sequence file for a single spin echo
    seq_file = constants.SCANNER_CONTROL_CAL_SEQ_FILES/'se_6.seq' # TODO: Seq file should be loadable or new sequence should be made

    # Run the experiment once to prep array
    rxd, rx_t = scr.run_pulseq(seq_file, rf_center=larmor_freq,
                               tx_t=1, grad_t=10, tx_warmup=100,
                               shim_x=shim_x, shim_y=shim_y, shim_z=shim_z,
                               grad_cal=False, save_np=False, save_mat=False, save_msgs=False, gui_test=gui_test)

    # Create array for storing data
    rx_arr = np.zeros((rxd.shape[0], steps), dtype=np.cdouble)
    rx_arr[:, 0] = rxd

    # Pause for spin recovery
    time.sleep(delay_s)

    # Repeat for each frequency after the first
    for i in range(1, steps):
        print(f'{swept_freqs[i]:.4f} MHz ({i}/{steps})')
        rx_arr[:, i], _ = scr.run_pulseq(seq_file, rf_center=swept_freqs[i],
                                         tx_t=1, grad_t=10, tx_warmup=100,
                                         shim_x=shim_x, shim_y=shim_y, shim_z=shim_z,
                                         grad_cal=False, save_np=False, save_mat=False, save_msgs=False,
                                         gui_test=gui_test)

        time.sleep(delay_s)

    # Find the frequency data with the largest maximum absolute value
    max_ind = np.argmax(np.max(np.abs(rx_arr), axis=0, keepdims=False))
    max_freq = swept_freqs[max_ind]
    print(f'Max frequency: {max_freq:.4f} MHz')

    # Plot figure
    if plot:
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(f'{steps}-step search around {step_search_center:.4f} MHz')
        axs[0].plot(np.real(rx_arr))
        axs[0].legend([f'{freq:.4f} MHz' for freq in swept_freqs])
        axs[0].set_title('Concatenated signal -- Real')
        axs[1].plot(np.abs(rx_arr))
        axs[1].set_title('Concatenated signal -- Magnitude')
        plt.show()

    # Output of useful data for visualization
    data_dict = {'rx_arr': rx_arr,
                 'rx_t': rx_t,
                 'larmor_freq': larmor_freq
                 }

    # Return the frequency that worked the best
    return max_freq, data_dict


def larmor_cal(larmor_start=cfg.LARMOR_FREQ, iterations=10, delay_s=1, echo_count=2,
               step_size=0.6, plot=False, shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z, gui_test=False):
    """
    Run a gradient descent search from a starting larmor frequency, optimizing to find the frequency
    with the most constant phase.

    Args:
        larmor_start (float): [MHz] Starting frequency for search, defaults to config LARMOR_FREQ
        iterations (int): Default 10, number of iterations in search
        delay_s (float): Default 1, delay in seconds for spin recovery
        echo_count (int): Default 2, must be 1-6. Number of spins in echo train.
        step_size (float): Scaling parameter for gradient search -- found success with 0.6, must be positive
        plot (bool): Default False, plot final data
        shim_x, shim_y, shim_z (float): Shim value, defaults to config SHIM_ values, must be less than 1 magnitude
        gui_test (bool): Default False, takes dummy data instead of actual data for GUI testing away from scanner

    Returns:
        float: Estimated larmor frequency in MHz
        dict: Dictionary of data
    """

    # Set larmor frequency for the first iteration
    larmor_freq = larmor_start

    # Echo count needs to be one that matches a seq file in saved calibration files
    if echo_count not in {1, 2, 3, 4, 5, 6}:
        print('Echo count needs to be an integer between 1 and 6')
        return -1

    # Load saved spin echo train seq file that has the right number of echoes
    seq_file = cfg.MGH_PATH + f'cal_seq_files/se_{echo_count}.seq'

    # Search for the right number of iterations
    for i in range(iterations):
        print(f'Iteration {i + 1}/{iterations}: {larmor_freq:.5f} MHz')

        # Run the experiment from seq file
        rxd, rx_t = scr.run_pulseq(seq_file, rf_center=larmor_freq,
                                   tx_t=1, grad_t=10, tx_warmup=100,
                                   shim_x=shim_x, shim_y=shim_y, shim_z=shim_z,
                                   grad_cal=False, save_np=False, save_mat=False, save_msgs=False, gui_test=gui_test)

        # Split echos for FFT
        rx_arr = np.reshape(rxd, (echo_count, -1))
        rx_count = rx_arr.shape[1]

        # Set up average and standard deviation arrays
        avgs = np.zeros(echo_count)
        stds = np.zeros(echo_count)

        # For each echo, calculate the average slope of the phase plot by:
        #   - Calculating the difference between every point in the middle third of the echo data
        #   - Cutting out the largest differences (representing phase wraps) by removing all changes above a certain size
        #   - Averaging from there
        for echo_n in range(echo_count):
            dphis = np.ediff1d(np.angle(rx_arr[echo_n, rx_count // 3:2 * (rx_count // 3)]))
            stds[echo_n] = np.std(dphis)
            ordered_dphis = dphis[np.argsort(np.abs(dphis))]
            large_change_ind = np.argmax(np.abs(np.ediff1d(np.abs(ordered_dphis))))
            dphi_vals = ordered_dphis[:large_change_ind - 1]
            avgs[echo_n] = np.mean(dphi_vals)

        # Find the average slopes across echoes, find expected change in larmor frequency from there
        dphi = np.mean(avgs)
        dw = dphi / (rx_t * np.pi)
        std = np.mean(stds)
        print(f'  Estimated frequency offset: {dw:.6f} MHz')
        print(f'  Spread (std): {std:.6f}')

        # Update larmor frequency
        larmor_freq += dw * step_size

        # Delay for spin recovery
        time.sleep(delay_s)

    # Run once more to check final frequency
    rxd, rx_t = scr.run_pulseq(seq_file, rf_center=larmor_freq,
                               tx_t=1, grad_t=1, tx_warmup=100,
                               shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z,
                               grad_cal=False, save_np=False, save_mat=False, save_msgs=False, gui_test=gui_test)

    # Split echos for FFT
    rx_arr = np.reshape(rxd, (echo_count, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(rx_arr, axis=0), axes=(0,))

    # Set up x-axis for FFT data
    fft_bw = 1 / (rx_t)
    fft_x = np.linspace(larmor_freq - fft_bw / 2, larmor_freq + fft_bw / 2, num=rx_fft.shape[0])

    # Announce results
    print(f'Calibrated Larmor frequency: {larmor_freq:.6f} MHz')
    if std >= 1:
        print(f"Didn't converge, try {fft_x[np.argmax(rx_fft[:, 0])]:.6f}")

    # Plot if needed
    if plot:
        fig, axs = plt.subplots(5, 1, constrained_layout=True)

        if std < 1:
            fig.suptitle(f'Larmor: {larmor_freq:.4f} MHz')
        else:
            fig.suptitle(f"Didn't converge -- Try eyeballing from bottom graph")

        axs[0].plot(np.real(rxd))
        axs[0].set_title('Concatenated signal -- Real')

        axs[1].plot(np.abs(rxd))
        axs[1].set_title('Concatenated signal -- Magnitude')
        axs[1].sharex(axs[0])
        axs[1].set_xlabel('Samples')

        axs[2].plot(np.arange(0, rx_count) * rx_t, np.angle(rx_arr))
        axs[2].set_title('Stacked signals -- Phase')
        axs[2].set_xlabel('Time (us)')

        axs[3].plot(fft_x, np.abs(rx_fft))
        axs[3].set_title('Stacked signals -- FFT')

        axs[4].plot(fft_x, np.mean(np.abs(rx_fft), axis=1, keepdims=False))
        axs[4].set_title('Averaged signals -- FFT')
        axs[4].sharex(axs[3])
        axs[4].set_xlabel('Frequency (MHz)')
        plt.show()

    # Data saved for visualization
    data_dict = {'rxd': rxd,
                 'rx_t': rx_t,
                 'trs': echo_count,
                 'larmor_freq': larmor_freq
                 }

    return larmor_freq, data_dict


def rf_max_cal(larmor_freq=cfg.LARMOR_FREQ, points=20, iterations=2, zoom_factor=2,
               shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z,
               tr_spacing=2, force_tr=False, first_max=False, smooth=True, plot=True, gui_test=False):
    """
    Calibrate RF maximum for pi/2 flip angle

    Args:
        larmor_freq (float): [MHz] Scanner larmor frequency
        points (int): Points to plot per iteration
        iterations (int): Iterations to focus in
        zoom_factor (float): About to zoom in by each iteration -- must be greater than 1
        shim_x, shim_y, shim_z (float): Shim value, defaults to config SHIM_ values, must be less than 1 magnitude
        tr_spacing (float): [us] Time between repetitions
        force_tr (bool): Default False, forces long TR times that would otherwise throw an error
        first_max (bool): Default False, changes search to find the first maximum instead of global
        smooth (bool): Default True, 3-wide running average on data
        plot (bool): Default False, plot final data

    Returns:
        float: Estimated RF max in Hz
        dict: Dictionary of data
    """
    # Select seq file for 2 spin echoes
    seq_file = cfg.MGH_PATH + f'cal_seq_files/se_2.seq'
    RF_PI2_DURATION = 50  # us, hardcoded from sequence

    # Make sure the TR units are right (in case someone puts in us rather than s)
    if (tr_spacing >= 30) and not force_tr:
        print('TR spacing is over 30 seconds! Set "force_tr" to True if this isn\'t a mistake. ')
        return -1

    # Needs to zoom in, not out
    assert (zoom_factor > 1)

    # Cap search values to not hit system limits
    rf_min, rf_max = 0.05, 0.95
    rf_max_val = 0

    # Run iterative search
    for it in range(iterations):
        # Generate search range
        rf_amp_vals = np.linspace(rf_min, rf_max, num=points, endpoint=True)
        rxd_list = []
        print(f'Iteration {it + 1}/{iterations}: {points} points from {rf_min:.2f} to {rf_max:.2f} fractional RF power')

        # Repeatedly run the experiment from seq file
        for i in range(points):
            # Cap rf value if needed for system
            adj_rf_max = max(cfg.RF_MAX * cfg.RF_PI2_FRACTION, 5000) / rf_amp_vals[i]
            rxd, rx_t = scr.run_pulseq(seq_file, rf_center=larmor_freq,
                                       tx_t=1, grad_t=10, tx_warmup=100,
                                       shim_x=shim_x, shim_y=shim_y, shim_z=shim_z, rf_max=adj_rf_max,
                                       grad_cal=False, save_np=False, save_mat=False, save_msgs=False,
                                       gui_test=gui_test)
            rxd_list.append(rxd)
            time.sleep(tr_spacing)

            # Print progress
            if (i + 1) % 5 == 0:
                print(f'Finished point {i + 1}/{points}...')

        # Reshape data to split echoes, ignore first echo due to measurement inconsistencies
        rx_arr = np.reshape(rxd_list, (points, 2, -1))[:, 1, :]
        # Measure maximums of each measurement
        peak_max_arr = np.max(np.abs(rx_arr), axis=1, keepdims=False)

        # Smooth out data with a rolling average
        if smooth:
            peak_max_arr = np.convolve(np.hstack((peak_max_arr[0:1], peak_max_arr[0:1],
                                                  peak_max_arr, peak_max_arr[-1:], peak_max_arr[-1:])),
                                       [1 / 3, 1 / 3, 1 / 3])[3:-3]

        # Pick out first maximum or absolute maximum
        dec_inds = np.where(peak_max_arr[:-1] >= peak_max_arr[1:])[0]
        if first_max and len(dec_inds) > 0:
            max_ind = dec_inds[0]
        else:
            max_ind = np.argmax(peak_max_arr)
        rf_max_val = rf_amp_vals[max_ind]

        # Plot if asked
        if plot and it < iterations - 1:
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            fig.suptitle(f'Iteration {it + 1}/{iterations}')
            axs[0].plot(np.abs(rx_arr).T)
            axs[0].set_title('Stacked signals -- Magnitude')
            axs[1].plot(rf_amp_vals, peak_max_arr)
            axs[1].plot(rf_max_val, peak_max_arr[max_ind], 'x')
            if first_max:
                axs[1].set_title(f'Max signals -- first max at {rf_max_val:.4f}')
            else:
                axs[1].set_title(f'Max signals -- global max at {rf_max_val:.4f}')
            plt.ion()
            plt.show()
            plt.draw()
            plt.pause(0.001)

        # Update range by zooming around max value
        rf_min = max(0.05, rf_max_val - zoom_factor ** (-1 * (it + 1)) / 2)
        rf_max = min(0.95, rf_max_val + zoom_factor ** (-1 * (it + 1)) / 2)

    # Calculate RF max in Hz
    est_rf_max = 0.25 / (RF_PI2_DURATION * rf_max_val) * 1e6
    print(f'Estimated RF max: {est_rf_max:.2f} Hz')
    print(f'{RF_PI2_DURATION}us pulse, pi/2 flip maxed at {rf_max_val * cfg.RF_MAX / est_rf_max:.4f} fractional power')

    # Plot if asked
    if plot:
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(f'Iteration {it + 1}/{iterations}')
        axs[0].plot(np.abs(rx_arr).T)
        axs[0].set_title('Stacked signals -- Magnitude')
        axs[1].plot(rf_amp_vals, peak_max_arr)
        axs[1].plot(rf_max_val, peak_max_arr[max_ind], 'x')
        if first_max:
            axs[1].set_title(f'Max signals -- first max at {rf_max_val:.4f}')
        else:
            axs[1].set_title(f'Max signals -- global max at {rf_max_val:.4f}')
        plt.ioff()
        plt.show()

    # Saved data for visualization
    data_dict = {'rxd': rxd,
                 'rx_arr': rx_arr,
                 'rx_t': rx_t,
                 'rxd_list': rxd_list,
                 'rf_max': est_rf_max
                 }

    return est_rf_max, data_dict


# TODO Add gui test functionality
# TODO Comment
def grad_max_cal(channel='x', phantom_width=10, larmor_freq=cfg.LARMOR_FREQ, calibration_power=0.8,
                 trs=3, tr_spacing=2e6, echo_duration=5000,
                 readout_duration=500, rx_period=25 / 3,
                 RF_PI2_DURATION=50, rf_max=cfg.RF_MAX,
                 trap_ramp_duration=50, trap_ramp_pts=5,
                 plot=True):
    """
    Calibrate gradient maximum using a phantom of known width

    Args:
        phantom_width (float): [mm] Phantom width
        larmor_freq (float): [MHz] Scanner larmor frequency
        calibration_power (float): [arb.] Fractional power to evaluate at
        trs (int): [arb.] Number of times to repeat for averaging
        tr_spacing (float): [us] Time between repetitions
        echo_duration (float): [us] Time between echo peaks
        readout_duration (float): [us] Readout window around echo peak
        rx_period (float): [us] Readout dwell time
        gradient_overshoot (float): [us] Amount of time to hold the readout gradient on for longer than readout_duration
        RF_PI2_DURATION (float): [us] RF pi/2 pulse duration
        rf_max (float): [Hz] System RF max
        plot (bool): Default False, plot final data

    Returns:
        float: Estimated gradient max in Hz/m
    """

    if channel not in {'x', 'y', 'z'}:
        print(f"Invalid channel '{channel}' -- Expected 'x', 'y', or 'z'")
        return -1

    rf_scaling = .25 / (rf_max * 1e-6 * RF_PI2_DURATION)
    readout_duration = readout_duration
    init_gpa = True
    rf_pi_duration = 2 * RF_PI2_DURATION

    def rf_wf(tstart, echo_idx):
        pi2_phase = 1  # x
        pi_phase = 1j  # y
        if echo_idx == 0:
            # do pi/2 pulse, then start pi pulse
            return np.array(
                [tstart + (echo_duration - RF_PI2_DURATION) / 2, tstart + (echo_duration + RF_PI2_DURATION) / 2,
                 tstart + echo_duration - rf_pi_duration / 2]), np.array([pi2_phase, 0, pi_phase]) * rf_scaling
        else:
            # finish RF pulse
            return np.array([tstart + rf_pi_duration / 2]), np.array([0])

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 2  # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1  # us, time to keep the TX gate on after an RF pulse ends

        if echo_idx == 0:
            # do pi/2 pulse, then start pi pulse
            return np.array([tstart + (echo_duration - RF_PI2_DURATION) / 2 - tx_gate_pre,
                             tstart + (echo_duration + RF_PI2_DURATION) / 2 + tx_gate_post,
                             tstart + echo_duration - rf_pi_duration / 2 - tx_gate_pre]), \
                   np.array([1, 0, 1])
        else:
            # finish pi pulse
            return np.array([tstart + rf_pi_duration / 2 + tx_gate_post]), np.array([0])

    def readout_grad_wf(tstart, echo_idx):
        if echo_idx == 0:
            return trap_cent(tstart + echo_duration * 3 / 4, calibration_power, readout_duration / 2,
                             trap_ramp_duration, trap_ramp_pts)
        else:
            return trap_cent(tstart + echo_duration / 2, calibration_power, readout_duration,
                             trap_ramp_duration, trap_ramp_pts)

    def readout_wf(tstart, echo_idx):
        if echo_idx == 0:
            return np.array([tstart]), np.array([0])  # keep on zero for pi2
        else:
            return np.array([tstart + (echo_duration - readout_duration) / 2,
                             tstart + (echo_duration + readout_duration) / 2]), np.array([1, 0])

    expt = ex.Experiment(lo_freq=larmor_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin))

    global_t = 20  # start the first TR at 20us

    for _ in range(trs):
        for echo in range(2):
            tx_t, tx_a = rf_wf(global_t, echo)
            tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo)
            readout_t, readout_a = readout_wf(global_t, echo)
            rx_gate_t, rx_gate_a = readout_wf(global_t, echo)

            readout_grad_t, readout_grad_a = readout_grad_wf(global_t, echo)
            global_t += echo_duration

            expt.add_flodict({
                'tx0': (tx_t, tx_a),
                'tx1': (tx_t, tx_a),
                'grad_vx': (readout_grad_t, readout_grad_a * (channel == 'x')),
                'grad_vy': (readout_grad_t, readout_grad_a * (channel == 'y')),
                'grad_vz': (readout_grad_t, readout_grad_a * (channel == 'z')),
                'rx0_en': (readout_t, readout_a),
                'rx1_en': (readout_t, readout_a),
                'tx_gate': (tx_gate_t, tx_gate_a),
                'rx_gate': (rx_gate_t, rx_gate_a),
            })

        global_t += tr_spacing

    rx, _ = expt.run()
    expt.close_server(True)
    expt._s.close()  # close socket

    rxd = rx['rx0']

    rx_arr = np.reshape(rxd, (trs, -1))
    rx_arr_av = np.average(rx_arr, axis=0)
    rxd_av = np.reshape(rx_arr_av, (-1))

    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rxd_av)))

    peaks, _ = sig.find_peaks(np.abs(rx_fft), width=2)
    peak_results = sig.peak_widths(np.abs(rx_fft), peaks, rel_height=0.95)
    max_peak = np.argmax(peak_results[0])
    fwhm = peak_results[0][max_peak]

    fft_scale = 1e6 / (rx_period * rx_fft.shape[0])  # [Hz/index] Need to know what the distance on x is

    fft_bw = 1 / (rx_period)
    fft_x = np.linspace(larmor_freq - fft_bw / 2, larmor_freq + fft_bw / 2, num=rx_fft.shape[0])

    hline = np.array([peak_results[1][max_peak], peak_results[2][max_peak], peak_results[3][max_peak]])
    hline[1:] = hline[1:] * (fft_x[1] - fft_x[0]) + fft_x[0]

    grad_max = fwhm * fft_scale / (phantom_width * 1e-3 * calibration_power)  # [Hz/m]
    print(f'Gradient value: {(grad_max * calibration_power * 1e-3):.4f} KHz/m')
    print(f'Estimated gradient max: {(grad_max * 1e-3):.4f} KHz/m')

    if plot:
        _, axs = plt.subplots(4, 1, constrained_layout=True)
        axs[0].plot(np.real(rxd))
        axs[0].set_title('Concatenated signals -- Real, Magnitude')

        axs[1].plot(np.abs(rxd))
        axs[1].sharex(axs[0])
        axs[1].set_xlabel('Samples')

        axs[2].plot(np.arange(0, len(rxd_av)) * rx_period, np.abs(rxd_av))
        axs[2].set_title('Averaged TRs -- Magnitude')
        axs[2].set_xlabel('Time (us)')

        axs[3].plot(fft_x, np.abs(rx_fft))
        axs[3].hlines(*hline, 'r')
        axs[3].set_title(f'FFT -- Magnitude ({(grad_max * 1e-3):.4f} KHz/m gradient max)')
        axs[3].set_xlabel('Frequency (MHz)')
        plt.show()

    return grad_max


def shim_cal(larmor_freq=cfg.LARMOR_FREQ, channel='x', range=0.01, shim_points=3, points=2, iterations=1, zoom_factor=2,
             shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z,
             tr_spacing=2, force_tr=False, first_max=False, smooth=True, plot=True, gui_test=False):
    """
    Calibrate RF maximum for pi/2 flip angle


    Args:
        larmor_freq (float): [MHz] Scanner larmor frequency
        points (int): Points to plot per iteration
        iterations (int): Iterations to focus in
        zoom_factor (float): About to zoom in by each iteration -- must be greater than 1
        shim_x, shim_y, shim_z (float): Shim value, defaults to config SHIM_ values, must be less than 1 magnitude
        tr_spacing (float): [us] Time between repetitions
        force_tr (bool): Default False, forces long TR times that would otherwise throw an error
        first_max (bool): Default False, changes search to find the first maximum instead of global
        smooth (bool): Default True, 3-wide running average on data
        plot (bool): Default False, plot final data

    Returns:
        float: Estimated RF max in Hz
        dict: Dictionary of data
    """

    if channel not in {'x', 'y', 'z'}:
        print(f"Invalid channel '{channel}' -- Expected 'x', 'y', or 'z'")
        return -1

    seq_file = cfg.MGH_PATH + f'cal_seq_files/spin_echo_1D_proj.seq'
    rxd_list = []

    if channel == 'x':
        shim_centre = shim_x
    elif channel == 'y':
        shim_centre = shim_y
    else:
        shim_centre = shim_z

    shim_range = np.linspace(-range / 2, range / 2, shim_points) + shim_centre
    for shim in shim_range:
        if channel == 'x':
            shim_x = shim
        elif channel == 'y':
            shim_y = shim
        else:
            shim_z = shim

        rxd, rx_t = scr.run_pulseq(seq_file, rf_center=larmor_freq,
                                   tx_t=1, grad_t=10, tx_warmup=100,
                                   shim_x=shim_x, shim_y=shim_y, shim_z=shim_z,
                                   grad_cal=False, save_np=False, save_mat=False, save_msgs=False, gui_test=gui_test)
        rxd_list.append(rxd)
        time.sleep(tr_spacing)

    plt.subplot(2, 1, 1)
    for rx in rxd_list:
        rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rx)))
        # plt.plot(np.abs(k))
        plt.plot(np.abs(rx_fft))

    plt.legend(shim_range)

    plt.subplot(2, 1, 2)
    for rx in rxd_list:
        plt.plot(np.abs(rx))

    plt.legend(shim_range)

    plt.show()
    # import pdb;pdb.set_trace()

    # if False:
    #     # Select seq file for 2 spin echoes
    #     seq_file = cfg.MGH_PATH + f'cal_seq_files/spin_echo_1D_proj.seq'
    #     RF_PI2_DURATION = 50  # us, hardcoded from sequence
    #
    #     # Make sure the TR units are right (in case someone puts in us rather than s)
    #     if (tr_spacing >= 30) and not force_tr:
    #         print('TR spacing is over 30 seconds! Set "force_tr" to True if this isn\'t a mistake. ')
    #         return -1
    #
    #     # Needs to zoom in, not out
    #     assert (zoom_factor > 1)
    #
    #     # Cap search values to not hit system limits
    #     rf_min, rf_max = 0.05, 0.95
    #     rf_max_val = 0
    #
    #     # Run iterative search
    #     for it in range(iterations):
    #         # Generate search range
    #         rf_amp_vals = np.linspace(rf_min, rf_max, num=points, endpoint=True)
    #         rxd_list = []
    #         print(
    #             f'Iteration {it + 1}/{iterations}: {points} points from {rf_min:.2f} to {rf_max:.2f} fractional RF power')
    #
    #         # Repeatedly run the experiment from seq file
    #         for i in range(points):
    #             # Cap rf value if needed for system
    #             adj_rf_max = max(cfg.RF_MAX * cfg.RF_PI2_FRACTION, 5000) / rf_amp_vals[i]
    #             rxd, rx_t = scr.run_pulseq(seq_file, rf_center=larmor_freq,
    #                                        tx_t=1, grad_t=10, tx_warmup=100,
    #                                        shim_x=shim_x, shim_y=shim_y, shim_z=shim_z, rf_max=adj_rf_max,
    #                                        grad_cal=False, save_np=False, save_mat=False, save_msgs=False,
    #                                        gui_test=gui_test)
    #             rxd_list.append(rxd)
    #             time.sleep(tr_spacing)
    #
    #             # Print progress
    #             if (i + 1) % 5 == 0:
    #                 print(f'Finished point {i + 1}/{points}...')
    #
    #         # Reshape data to split echoes, ignore first echo due to measurement inconsistencies
    #         rx_arr = np.reshape(rxd_list, (points, 2, -1))[:, 1, :]
    #         # Measure maximums of each measurement
    #         peak_max_arr = np.max(np.abs(rx_arr), axis=1, keepdims=False)
    #
    #         # Smooth out data with a rolling average
    #         if smooth:
    #             peak_max_arr = np.convolve(np.hstack((peak_max_arr[0:1], peak_max_arr[0:1],
    #                                                   peak_max_arr, peak_max_arr[-1:], peak_max_arr[-1:])),
    #                                        [1 / 3, 1 / 3, 1 / 3])[3:-3]
    #
    #         # Pick out first maximum or absolute maximum
    #         dec_inds = np.where(peak_max_arr[:-1] >= peak_max_arr[1:])[0]
    #         if first_max and len(dec_inds) > 0:
    #             max_ind = dec_inds[0]
    #         else:
    #             max_ind = np.argmax(peak_max_arr)
    #         rf_max_val = rf_amp_vals[max_ind]
    #
    #         # Plot if asked
    #         if plot and it < iterations - 1:
    #             fig, axs = plt.subplots(2, 1, constrained_layout=True)
    #             fig.suptitle(f'Iteration {it + 1}/{iterations}')
    #             axs[0].plot(np.abs(rx_arr).T)
    #             axs[0].set_title('Stacked signals -- Magnitude')
    #             axs[1].plot(rf_amp_vals, peak_max_arr)
    #             axs[1].plot(rf_max_val, peak_max_arr[max_ind], 'x')
    #             if first_max:
    #                 axs[1].set_title(f'Max signals -- first max at {rf_max_val:.4f}')
    #             else:
    #                 axs[1].set_title(f'Max signals -- global max at {rf_max_val:.4f}')
    #             plt.ion()
    #             plt.show()
    #             plt.draw()
    #             plt.pause(0.001)
    #
    #         # Update range by zooming around max value
    #         rf_min = max(0.05, rf_max_val - zoom_factor ** (-1 * (it + 1)) / 2)
    #         rf_max = min(0.95, rf_max_val + zoom_factor ** (-1 * (it + 1)) / 2)
    #
    #     # Calculate RF max in Hz
    #     est_rf_max = 0.25 / (RF_PI2_DURATION * rf_max_val) * 1e6
    #     print(f'Estimated RF max: {est_rf_max:.2f} Hz')
    #     print(
    #         f'{RF_PI2_DURATION}us pulse, pi/2 flip maxed at {rf_max_val * cfg.RF_MAX / est_rf_max:.4f} fractional power')
    #
    #     # Plot if asked
    #     if plot:
    #         fig, axs = plt.subplots(2, 1, constrained_layout=True)
    #         fig.suptitle(f'Iteration {it + 1}/{iterations}')
    #         axs[0].plot(np.abs(rx_arr).T)
    #         axs[0].set_title('Stacked signals -- Magnitude')
    #         axs[1].plot(rf_amp_vals, peak_max_arr)
    #         axs[1].plot(rf_max_val, peak_max_arr[max_ind], 'x')
    #         if first_max:
    #             axs[1].set_title(f'Max signals -- first max at {rf_max_val:.4f}')
    #         else:
    #             axs[1].set_title(f'Max signals -- global max at {rf_max_val:.4f}')
    #         plt.ioff()
    #         plt.show()
    #
    #     # Saved data for visualization
    #     data_dict = {'rxd': rxd,
    #                  'rx_arr': rx_arr,
    #                  'rx_t': rx_t,
    #                  'rxd_list': rxd_list,
    #                  'rf_max': est_rf_max
    #                  }
    #
    #     return est_rf_max, data_dict


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        command = sys.argv[1]

        if command == 'larmor':
            if len(sys.argv) == 3:
                larmor_cal(plot=True, echo_count=int(sys.argv[2]), gui_test=False)
            else:
                larmor_cal(plot=True, gui_test=False)
        elif command == 'larmor_w':
            if len(sys.argv) == 3:
                start_freq, _ = larmor_step_search(plot=True, steps=int(sys.argv[2]))
            else:
                start_freq, _ = larmor_step_search(plot=True)
            larmor_cal(larmor_start=start_freq, plot=True)
        elif command == 'rf':
            rf_max_cal(plot=True)
        elif command == 'grad':
            if len(sys.argv) == 3:
                grad_max_cal(channel=sys.argv[2], plot=True)
            else:
                grad_max_cal(plot=True)
        else:
            print('Enter a calibration command from: [larmor, larmor_w, rf, grad]')
    else:
        print('Enter a calibration command from: [larmor, larmor_w, rf, grad]')
