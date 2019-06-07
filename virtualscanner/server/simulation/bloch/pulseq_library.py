from math import pi

import numpy as np

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import makeadc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc import make_sinc_pulse
from pypulseq.make_trap import make_trapezoid
from pypulseq.opts import Opts

GAMMA_BAR = 42.5775e6
GAMMA = 2 * pi * GAMMA_BAR


# TODO make these multi-slice sequences, and modify recon in simulator!!!

def make_pulseq_gre(fov, n, thk, fa, tr, te, enc='xyz', slice_locs=None, write=False):
    """
    Makes a single-slice GRE pulseq sequence
    INPUTS
    fov - field-of-view (m)
    n - matrix size
    thk - slice thickness (m)
    fa - flip angle (deg)
    tr - repetition time (s)
    te - echo time (s)
    enc - encoding. 1st-readout; 2nd-phase enc.; 3rd-slice select
          e.g. 'xyz': axial(z) slice, readout in x and phase enc. steps in y
    loc - slice location (off-center; m)
    write - whether to write seq into file; default is yes

    OUTPUT
    seq - pulseq Sequence object
    """
    kwargs_for_opts = {"rf_ring_down_time": 0, "rf_dead_time": 0}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)

    Nf = n
    Np = n
    flip = fa * pi / 180
    kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 4e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf, g_ss = make_sinc_pulse(kwargs_for_sinc, 2)
    g_ss.channel = enc[2]

    delta_k = 1 / fov
    kWidth = Nf * delta_k

    # Readout and ADC
    readoutTime = 6.4e-3
    kwargs_for_g_ro = {"channel": enc[0], "system": system, "flat_area": kWidth, "flat_time": readoutTime}
    g_ro = make_trapezoid(kwargs_for_g_ro)
    kwargs_for_adc = {"num_samples": Nf, "duration": g_ro.flat_time, "delay": g_ro.rise_time}
    adc = makeadc(kwargs_for_adc)
    # Readout rewinder
    kwargs_for_g_ro_pre = {"channel": enc[0], "system": system, "area": -g_ro.area / 2, "duration": 2e-3}
    g_ro_pre = make_trapezoid(kwargs_for_g_ro_pre)
    # Slice refocusing
    kwargs_for_g_ss_reph = {"channel": enc[2], "system": system, "area": -g_ss.area / 2, "duration": 2e-3}
    g_ss_reph = make_trapezoid(kwargs_for_g_ss_reph)
    phase_areas = (np.arange(Np) - (Np / 2)) * delta_k

    # TE, TR = 10e-3, 1000e-3
    TE, TR = te, tr
    delayTE = TE - calc_duration(g_ro_pre) - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2
    delayTR = TR - calc_duration(g_ro_pre) - calc_duration(g_ss) - calc_duration(g_ro) - delayTE
    delay1 = make_delay(delayTE)
    delay2 = make_delay(delayTR)

    if slice_locs is None:
        locs = [0]
    else:
        locs = slice_locs

    for u in range(len(locs)):
        # add frequency offset
        rf.freq_offset = g_ss.amplitude * locs[u]
        for i in range(Np):
            seq.add_block(rf, g_ss)
            kwargs_for_g_pe = {"channel": enc[1], "system": system, "area": phase_areas[i], "duration": 2e-3}
            g_pe = make_trapezoid(kwargs_for_g_pe)
            seq.add_block(g_ro_pre, g_pe, g_ss_reph)
            seq.add_block(delay1)
            seq.add_block(g_ro, adc)
            seq.add_block(delay2)

    if write:
        seq.write(
            "gre_fov{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:.0f}deg.seq".format(fov * 1000, Nf, Np, TE * 1000,
                                                                                         TR * 1000, flip * 180 / pi))
    print('GRE sequence constructed')
    return seq


def make_pulseq_irse(fov, n, thk, fa, tr, te, ti, enc='xyz', slice_locs=None, write=False):
    """
    Makes a single-slice IRSE pulseq sequence
    INPUTS
    fov - field-of-view (m)
    n - matrix size
    thk - slice thickness (m)
    fa - flip angle (deg)
    tr - repetition time (s)
    te - echo time (s)
    ti - inversion time(s) (s):
         if multiple, input as an array [ti1,ti2,ti3,...]
         if only 1 TI, input float or int
    enc - encoding. 1st-readout; 2nd-phase enc.; 3rd-slice select
          e.g. 'xyz': axial(z) slice, readout in x and phase enc. steps in y
    loc - slice location (off-center; m)
    write - whether to write seq into file; default is yes

    OUTPUT
    seq - pulseq Sequence object
    """
    kwargs_for_opts = {"max_grad": 33, "grad_unit": "mT/m", "max_slew": 100, "slew_unit": "T/m/s",
                       "rf_dead_time": 10e-6,
                       "adc_dead_time": 10e-6}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)
    # Parameters
    Nf = n
    Np = n
    delta_k = 1 / fov
    kWidth = Nf * delta_k

    TI, TE, TR = ti, te, tr

    if np.shape(TI) == ():
        TI = [TI]

    # Non-180 pulse
    flip1 = fa * pi / 180
    kwargs_for_sinc = {"flip_angle": flip1, "system": system, "duration": 2e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf, g_ss = make_sinc_pulse(kwargs_for_sinc, 2)
    g_ss.channel = enc[2]

    # 180 pulse
    flip2 = 180 * pi / 180
    kwargs_for_sinc = {"flip_angle": flip2, "system": system, "duration": 2e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf180, g_ss180 = make_sinc_pulse(kwargs_for_sinc, 2)
    g_ss180.channel = enc[2]

    # Readout gradient & ADC
    # readoutTime = system.grad_raster_time * Nf
    readoutTime = 6.4e-3

    kwargs_for_g_ro = {"channel": enc[0], "system": system, "flat_area": kWidth, "flat_time": readoutTime}
    g_ro = make_trapezoid(kwargs_for_g_ro)
    kwargs_for_adc = {"num_samples": Nf, "system": system, "duration": g_ro.flat_time, "delay": g_ro.rise_time}
    adc = makeadc(kwargs_for_adc)

    # RO rewinder gradient
    kwargs_for_g_ro_pre = {"channel": enc[0], "system": system, "area": g_ro.area / 2,
                           "duration": 2e-3}

    g_ro_pre = make_trapezoid(kwargs_for_g_ro_pre)

    # Slice refocusing gradient
    kwargs_for_g_ss_reph = {"channel": enc[2], "system": system, "area": -g_ss.area / 2, "duration": 2e-3}
    g_ss_reph = make_trapezoid(kwargs_for_g_ss_reph)

    # Delays # TODO timing problem!!??

    delayTE1 = TE / 2 - max(calc_duration(g_ss_reph), calc_duration(g_ro_pre)) - calc_duration(
        g_ss) / 2 - calc_duration(
        g_ss180) / 2
    delayTE2 = TE / 2 - calc_duration(g_ro) / 2 - calc_duration(g_ss180) / 2
    delayTE3 = TR - TE - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2

    print('dur rf', calc_duration(rf), 'dur gss:', calc_duration(g_ss))

    delay1 = make_delay(delayTE1)
    delay2 = make_delay(delayTE2)
    delay3 = make_delay(delayTE3)

    # Construct sequence
    if slice_locs is None:
        locs = [0]
    else:
        locs = slice_locs

    for inv in range(len(TI)):
        for u in range(len(locs)):
            rf180.freq_offset = g_ss180.amplitude * locs[u]
            rf.freq_offset = g_ss.amplitude * locs[u]
            for i in range(Np):
                # Inversion Recovery part
                seq.add_block(rf180)  # Non-selective at the moment; could be extended to make this selective/adiabatic
                seq.add_block(
                    make_delay(TI[inv] - calc_duration(rf) / 2 - calc_duration(rf180) / 2))  # Inversion time delay
                # Spin echo part
                seq.add_block(rf, g_ss)  # 90-deg pulse
                kwargs_for_g_pe_pre = {"channel": enc[1], "system": system, "area": -(Np / 2 - i) * delta_k,
                                       "duration": 2e-3}
                g_pe_pre = make_trapezoid(kwargs_for_g_pe_pre)  # Phase encoding gradient
                seq.add_block(g_ro_pre, g_pe_pre,
                              g_ss_reph)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
                seq.add_block(delay1)  # Delay 1: until 180-deg pulse
                seq.add_block(rf180, g_ss180)  # 180 deg pulse for SE
                seq.add_block(delay2)  # Delay 2: until readout
                seq.add_block(g_ro, adc)  # Readout!
                seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        if len(TI) == 1:
            seq.write("irse_fov{:.0f}mm_Nf{:d}_Np{:d}_TI{:.0f}ms_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nf, Np,
                                                                                                   TI[0] * 1000,
                                                                                                   TE * 1000,
                                                                                                   TR * 1000))
        else:
            seq.write(
                "irse_fov{:.0f}mm_Nf{:d}_Np{:d}_multiTI_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nf, Np, TE * 1000,
                                                                                          TR * 1000))

    print('IRSE sequence constructed')
    return seq


def make_pulseq_se(fov, n, thk, fa, tr, te, enc='xyz', slice_locs=None, write=False):
    """
    Makes a single-slice IRSE pulseq sequence
    INPUTS
    fov - field-of-view (m)
    n - matrix size
    thk - slice thickness (m)
    fa - flip angle (deg)
    tr - repetition time (s)
    te - echo time (s)
    enc - encoding. 1st-readout; 2nd-phase enc.; 3rd-slice select
          e.g. 'xyz': axial(z) slice, readout in x and phase enc. steps in y
    loc - slice location (off-center; m)
    write - whether to write seq into file; default is yes

    OUTPUT
    seq - pulseq Sequence object
    """
    kwargs_for_opts = {"max_grad": 33, "grad_unit": "mT/m", "max_slew": 100, "slew_unit": "T/m/s",
                       "rf_dead_time": 10e-6,
                       "adc_dead_time": 10e-6}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)
    # Parameters
    Nf = n
    Np = n
    delta_k = 1 / fov
    kWidth = Nf * delta_k

    TE, TR = te, tr

    # Non-180 pulse
    flip1 = fa * pi / 180
    kwargs_for_sinc = {"flip_angle": flip1, "system": system, "duration": 2e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf, g_ss = make_sinc_pulse(kwargs_for_sinc, 2)
    g_ss.channel = enc[2]

    # 180 pulse
    flip2 = 180 * pi / 180
    kwargs_for_sinc = {"flip_angle": flip2, "system": system, "duration": 2e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf180, g_ss180 = make_sinc_pulse(kwargs_for_sinc, 2)
    g_ss180.channel = enc[2]

    # Readout gradient & ADC
    #    readoutTime = system.grad_raster_time * Nf
    readoutTime = 6.4e-3
    kwargs_for_g_ro = {"channel": enc[0], "system": system, "flat_area": kWidth, "flat_time": readoutTime}
    g_ro = make_trapezoid(kwargs_for_g_ro)
    kwargs_for_adc = {"num_samples": Nf, "system": system, "duration": g_ro.flat_time, "delay": g_ro.rise_time}
    adc = makeadc(kwargs_for_adc)

    # RO rewinder gradient
    kwargs_for_g_ro_pre = {"channel": enc[0], "system": system, "area": g_ro.area / 2,
                           "duration": 2e-3}
    #                        "duration": g_ro.rise_time + g_ro.fall_time + readoutTime / 2}

    g_ro_pre = make_trapezoid(kwargs_for_g_ro_pre)

    # Slice refocusing gradient
    kwargs_for_g_ss_reph = {"channel": enc[2], "system": system, "area": -g_ss.area / 2, "duration": 2e-3}
    g_ss_reph = make_trapezoid(kwargs_for_g_ss_reph)

    # Delays
    delayTE1 = (TE - 2 * max(calc_duration(g_ss_reph), calc_duration(g_ro_pre)) - calc_duration(g_ss) - calc_duration(
        g_ss180)) / 2
    # delayTE2 = TE / 2 - calc_duration(g_ro) / 2 - calc_duration(g_ss180) / 2
    delayTE2 = (TE - calc_duration(g_ro) - calc_duration(g_ss180)) / 2
    delayTE3 = TR - TE - (calc_duration(g_ss) + calc_duration(g_ro)) / 2

    delay1 = make_delay(delayTE1)
    delay2 = make_delay(delayTE2)
    delay3 = make_delay(delayTE3)

    # Construct sequence
    if slice_locs is None:
        locs = [0]
    else:
        locs = slice_locs

    for u in range(len(locs)):
        rf180.freq_offset = g_ss180.amplitude * locs[u]
        rf.freq_offset = g_ss.amplitude * locs[u]
        for i in range(Np):
            seq.add_block(rf, g_ss)  # 90-deg pulse
            kwargs_for_g_pe_pre = {"channel": enc[1], "system": system, "area": -(Np / 2 - i) * delta_k,
                                   "duration": 2e-3}
            # "duration": g_ro.rise_time + g_ro.fall_time + readoutTime / 2}
            g_pe_pre = make_trapezoid(kwargs_for_g_pe_pre)  # Phase encoding gradient
            seq.add_block(g_ro_pre, g_pe_pre,
                          g_ss_reph)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
            seq.add_block(delay1)  # Delay 1: until 180-deg pulse
            seq.add_block(rf180, g_ss180)  # 180 deg pulse for SE
            seq.add_block(delay2)  # Delay 2: until readout
            seq.add_block(g_ro, adc)  # Readout!
            seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        seq.write(
            "se_fov{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nf, Np, TE * 1000, TR * 1000))

    print('Spin echo sequence constructed')
    return seq
