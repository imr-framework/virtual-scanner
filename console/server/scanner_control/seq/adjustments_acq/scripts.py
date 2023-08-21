#!/usr/bin/env python3
#
# Run a pulseq file
# Code by Lincoln Craven-Brightman
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig
from operator import itemgetter
import sys
import os
import console.server.scanner_control.seq.adjustments_acq.config as cfg  # pylint: disable=import-error
import console.server.scanner_control.seq.marcos_client.experiment as ex  # pylint: disable=import-error
from console.server.scanner_control.seq.flocra_pulseq.interpreter import PSInterpreter  # pylint: disable=import-error
from console.utils import constants

def run_pulseq(seq_file, rf_center=cfg.LARMOR_FREQ, rf_max=cfg.RF_MAX,
               gx_max=cfg.GX_MAX, gy_max=cfg.GY_MAX, gz_max=cfg.GZ_MAX,
               tx_t=1, grad_t=10, tx_warmup=100,
               shim_x=cfg.SHIM_X, shim_y=cfg.SHIM_Y, shim_z=cfg.SHIM_Z,
               grad_cal=False, save_np=False, save_mat=False, save_msgs=False,
               expt=None, plot_instructions=False, gui_test=False):
    """
    Interpret pulseq .seq file through flocra_pulseq

    Args:
        seq_file (string): Pulseq file in mgh.config SEQ file directory
        rf_center (float): [MHz] Center for frequency (larmor)
        rf_max (float): [Hz] Maximum system RF value for instruction scaling
        g[x, y, z]_max (float): [Hz/m] Maximum system gradient values for x, y, z for instruction scaling
        tx_t (float): [us] Raster period for transmit
        grad_t (float): [us] Raster period for gradients
        tx_warmup (float): [us] Warmup time for transmit gate, used to check pulseq file
        shim_x, shim_y, shim_z (float): Shim value, defaults to config SHIM_ values, must be less than 1 magnitude
        grad_cal (bool): Default False, run GPA_FHDO gradient calibration
        save_np (bool): Default False, save data in .npy format in mgh.config DATA directory
        save_mat (bool): Default False, save data in .mat format in mgh.config DATA directory
        save_msgs (bool): Default False, save log messages in .txt format in mgh.config DATA directory
        expt (flocra_pulseq.interpreter): Default None, pass in existing experiment to continue an object
        plot_instructions (bool): Default None, plot instructions for debugging
        gui_test (bool): Default False, load dummy data for gui testing

    Returns:
        numpy.ndarray: Rx data array
        float: (us) Rx period
    """

    # Load dummy data for GUI testing
    if gui_test:
        return np.load(cfg.MGH_PATH + 'test_data/gui.npy'), 25 / 3

    # Convert .seq file to machine dict
    psi = PSInterpreter(rf_center=rf_center * 1e6,
                        tx_warmup=tx_warmup,
                        rf_amp_max=rf_max,
                        tx_t=tx_t,
                        grad_t=grad_t,
                        gx_max=gx_max,
                        gy_max=gy_max,
                        gz_max=gz_max,
                        log_file=constants.SCANNER_CONTROL_LOG/'ps-interpreter.log')
    instructions, param_dict = psi.interpret(seq_file)

    # Shim
    instructions = shim(instructions, (shim_x, shim_y, shim_z))

    # Initialize experiment class
    if expt is None:
        expt = ex.Experiment(lo_freq=rf_center,
                             rx_t=param_dict['rx_t'],
                             init_gpa=True,
                             gpa_fhdo_offset_time=grad_t / 3,
                             halt_and_reset=True)

    # Optionbally run gradient linearization calibration
    if grad_cal:
        expt.gradb.calibrate(channels=[0, 1, 2], max_current=1, num_calibration_points=30, averages=5, poly_degree=5)

    # Add flat delay to avoid housekeeping at the start
    flat_delay = 10
    for buf in instructions.keys():
        instructions[buf] = (instructions[buf][0] + flat_delay, instructions[buf][1])

    # Plot instructions if needed
    if plot_instructions:
        _, axs = plt.subplots(len(instructions), 1, constrained_layout=True)
        for i, key in enumerate(instructions.keys()):
            axs[i].step(instructions[key][0], instructions[key][1], where='post')
            axs[i].plot(instructions[key][0], instructions[key][1], 'rx')
            axs[i].set_title(key)
        plt.show()

    # Load instructions
    expt.add_flodict(instructions)

    # Run experiment
    rxd, msgs = expt.run()

    # Optionally save messages
    if save_msgs:
        print(msgs)  # TODO include message saving

    # Announce completion
    nSamples = param_dict['readout_number']
    print(f'Finished -- read {nSamples} samples')

    # Optionally save rx output array as .npy file
    if save_np:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%y-%d-%m %H_%M_%S")
        filename = cfg.DATA_PATH + f"/{current_time}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        np.save(filename, rxd['rx0'])

    # Optionally save rx output array as .mat file
    if save_mat:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%y-%d-%m %H_%M_%S")
        filename = cfg.DATA_PATH + f"/{current_time}.mat"
        if os.path.exists(filename):
            os.remove(filename)
        sio.savemat(filename, {'flocra_data': rxd['rx0']})

    expt.__del__()

    # Return rx output array and rx period
    return rxd['rx0'], param_dict['rx_t']


def shim(instructions, shim):
    """
    Modify gradient instructions to shim the gradients for the experiment

    Args:
        instructions (dict): Instructions to modify
        shim (tuple): X, Y, Z shim values to use

    Returns:
        dict: Shimmed instructions
    """
    grads = ['grad_vx', 'grad_vy', 'grad_vz']
    for ch in range(3):
        updates = instructions[grads[ch]][1]
        updates[:-1] = updates[:-1] + shim[ch]
        assert (np.all(np.abs(updates) <= 1)), (f'Shim {shim[ch]} was too large for {grads[ch]}: '
                                                + f'{updates[np.argmax(np.abs(updates))]}')
        instructions[grads[ch]] = (instructions[grads[ch]][0], updates)
    return instructions


def recon_0d(rxd, rx_t, trs=1, larmor_freq=cfg.LARMOR_FREQ):
    """
    Reconstruct FFT data, pass data out to plotting or saving programs

    Args:
        rxd (numpy.ndarray): Rx data array
        rx_t (float): [us] Rx sample period
        trs (int): Number of repetitions to split apart
        larmor_freq (float): [MHz] Larmor frequency of data for FFT

    Returns:
        dict: Useful reconstructed data dictionary
    """
    # Split echos for FFT
    rx_arr = np.reshape(rxd, (trs, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rx_arr, axes=(0,)), axis=0), axes=(0,))
    x = np.linspace(0, rx_arr.shape[0] * rx_t * 1e-6, num=rx_arr.shape[0], endpoint=False)

    fft_bw = 1 / (rx_t)
    fft_x = np.linspace(larmor_freq - fft_bw / 2, larmor_freq + fft_bw / 2, num=rx_fft.shape[0])
    out_dict = {'dim': 0,
                'rxd': rxd,
                'rx_t': rx_t,
                'trs': trs,
                'rx_arr': rx_arr,
                'rx_fft': rx_fft,
                'x': x,
                'fft_bw': fft_bw,
                'fft_x': fft_x,
                'larmor_freq': larmor_freq
                }
    return out_dict


def recon_1d(rxd, rx_t, trs=1, larmor_freq=cfg.LARMOR_FREQ):
    """
    Reconstruct 1D data, pass data out to plotting or saving programs

    Args:
        rxd (numpy.ndarray): Rx data array
        rx_t (float): [us] Rx sample period
        trs (int): Number of repetitions to split apart
        larmor_freq (float): [MHz] Larmor frequency of data for FFT

    Returns:
        dict: Useful reconstructed data dictionary
    """
    # Split echos for FFT
    rx_arr = np.reshape(rxd, (trs, -1)).T
    rx_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rx_arr, axes=(0,)), axis=0), axes=(0,))
    x = np.linspace(0, rx_arr.shape[0] * rx_t * 1e-6, num=rx_arr.shape[0], endpoint=False)

    fft_bw = 1 / (rx_t)
    fft_x = np.linspace(larmor_freq - fft_bw / 2, larmor_freq + fft_bw / 2, num=rx_fft.shape[0])
    out_dict = {'dim': 1,
                'rxd': rxd,
                'rx_t': rx_t,
                'trs': trs,
                'rx_arr': rx_arr,
                'rx_fft': rx_fft,
                'x': x,
                'fft_bw': fft_bw,
                'fft_x': fft_x,
                'larmor_freq': larmor_freq
                }
    return out_dict


def recon_2d(rxd, trs, larmor_freq=cfg.LARMOR_FREQ):
    """
    Reconstruct 2D data, pass data out to plotting or saving programs

    Args:
        rxd (numpy.ndarray): Rx data array
        trs (int): Number of repetitions (Phase encode direction length)
        larmor_freq (float): [MHz] Larmor frequency of data for FFT

    Returns:
        dict: Useful reconstructed data dictionary
    """
    rx_arr = np.reshape(rxd, (trs, -1))
    rx_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rx_arr)))
    out_dict = {'dim': 2,
                'rxd': rxd,
                'trs': trs,
                'rx_arr': rx_arr,
                'rx_fft': rx_fft,
                'larmor_freq': larmor_freq
                }
    return out_dict


def peak_width_1d(recon_dict):
    """
    Find peak width from reconstructed data

    Args:
        recon_dict (dict): Reconstructed data dictionary

    Returns:
        dict: Line info dictionary
    """
    rx_fft, fft_x = itemgetter('rx_fft', 'fft_x')(recon_dict)

    peaks, _ = sig.find_peaks(np.abs(rx_fft), width=2)
    peak_results = sig.peak_widths(np.abs(rx_fft), peaks, rel_height=0.95)
    max_peak = np.argmax(peak_results[0])
    fwhm = peak_results[0][max_peak]

    hline = np.array([peak_results[1][max_peak], peak_results[2][max_peak], peak_results[3][max_peak]])
    hline[1:] = hline[1:] * (fft_x[1] - fft_x[0]) + fft_x[0]
    out_dict = {'hline': hline,
                'fwhm': fwhm,
                }
    return out_dict


def plot_signal_1d(recon_dict):
    # Example plotting function
    # Split echos for FFT
    x, rxd, rx_arr, rx_fft, fft_x = itemgetter('x', 'rxd', 'rx_arr', 'rx_fft', 'fft_x')(recon_dict)

    _, axs = plt.subplots(4, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(x, np.angle(rx_arr))
    axs[2].set_title('Stacked signals -- Phase')
    axs[3].plot(fft_x, np.abs(rx_fft))
    axs[3].set_title('Stacked signals -- FFT')
    plt.show()


def plot_signal_2d(recon_dict):
    # Example plotting function
    rx_arr, rx_fft, rxd = itemgetter('rx_arr', 'rx_fft', 'rxd')(recon_dict)

    _, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(np.real(rxd))
    axs[0].set_title('Concatenated signal -- Real')
    axs[1].plot(np.abs(rxd))
    axs[1].set_title('Concatenated signal -- Magnitude')
    axs[2].plot(np.abs(rx_arr))
    axs[2].set_title('Stacked signals -- Magnitude')
    fig, im_axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle('2D Image')
    im_axs[0].imshow(np.abs(rx_fft), cmap=plt.cm.bone)
    im_axs[0].set_title('Magnitude')
    im_axs[1].imshow(np.angle(rx_fft))
    im_axs[1].set_title('Phase')
    plt.show()


if __name__ == "__main__":
    # Maybe clean up
    if len(sys.argv) >= 2:
        command = sys.argv[1]
        if command == 'pulseq':
            if len(sys.argv) == 3:
                seq_file = cfg.SEQ_PATH + sys.argv[2]
                _, rx_t = run_pulseq(seq_file, save_np=True, save_mat=True)
                print(f'rx_t = {rx_t}')
            else:
                print(
                    '"pulseq" takes one .seq filename as an argument (just the filename, make sure it\'s in your seq_files path!)')
        elif command == 'plot2d':
            if len(sys.argv) == 4:
                rxd = np.load(cfg.DATA_PATH + sys.argv[2])
                tr_count = int(sys.argv[3])
                plot_signal_2d(recon_2d(rxd, tr_count, larmor_freq=cfg.LARMOR_FREQ))
            else:
                print('Format arguments as "plot2d [2d_data_filename] [tr count]"')
        elif command == 'plot1d':
            if len(sys.argv) == 5:
                rxd = np.load(cfg.DATA_PATH + sys.argv[2])
                rx_t = float(sys.argv[3])
                tr_count = int(sys.argv[4])
                plot_signal_1d(recon_1d(rxd, rx_t, trs=tr_count))
            else:
                print('Format arguments as "plot1d [1d_data_filename] [rx_t] [tr_count]"')
        elif command == 'plot_se':
            if len(sys.argv) == 5:
                rxd = np.load(cfg.DATA_PATH + sys.argv[2])
                rx_t = float(sys.argv[3])
                tr_count = int(sys.argv[4])
                plot_signal_1d(recon_0d(rxd, rx_t, trs=tr_count))
            else:
                print('Format arguments as "plot_se [spin_echo_data_filename] [rx_t] [tr_count]"')

        else:
            print('Enter a script command from: [pulseq, plot_se, plot1d, plot2d]')
    else:
        print('Enter a script command from: [pulseq, plot_se, plot1d, plot2d]')
