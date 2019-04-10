from math import pi

import numpy as np

from pulseq.core.Sequence.sequence import Sequence
from pulseq.core.calc_duration import calc_duration
from pulseq.core.make_adc import makeadc
from pulseq.core.make_delay import make_delay
from pulseq.core.make_sinc import make_sinc_pulse
from pulseq.core.make_trap import make_trapezoid
from pulseq.core.opts import Opts


def make_pulseq_gre(fov,n,thk,fa,tr,te,write=True):
    kwargs_for_opts = {"rf_ring_down_time": 0, "rf_dead_time": 0}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)

    Nx = n
    Ny = n
    flip = fa * pi / 180
    kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 4e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)
    # plt.plot(rf.t[0], rf.signal[0])
    # plt.show()

    delta_k = 1 / fov
    kWidth = Nx * delta_k
    readoutTime = 6.4e-3
    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": kWidth, "flat_time": readoutTime}
    gx = make_trapezoid(kwargs_for_gx)
    kwargs_for_adc = {"num_samples": Nx, "duration": gx.flat_time, "delay": gx.rise_time}
    adc = makeadc(kwargs_for_adc)

    kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -gx.area / 2, "duration": 2e-3}
    gx_pre = make_trapezoid(kwargs_for_gxpre)
    kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": 2e-3}
    gz_reph = make_trapezoid(kwargs_for_gz_reph)
    phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k

    # TE, TR = 10e-3, 1000e-3
    TE, TR = te,tr
    delayTE = TE - calc_duration(gx_pre) - calc_duration(gz) / 2 - calc_duration(gx) / 2
    delayTR = TR - calc_duration(gx_pre) - calc_duration(gz) - calc_duration(gx) - delayTE
    delay1 = make_delay(delayTE)
    delay2 = make_delay(delayTR)

    for i in range(Ny):
        seq.add_block(rf, gz)
        kwargsForGyPre = {"channel": 'y', "system": system, "area": phase_areas[i], "duration": 2e-3}
        gyPre = make_trapezoid(kwargsForGyPre)
        seq.add_block(gx_pre, gyPre, gz_reph)
        seq.add_block(delay1)
        seq.add_block(gx,adc)
        seq.add_block(delay2)

    if write:
        seq.write("gre_fov{:.0f}mm_Nx{:d}_Ny{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:.0f}deg.seq".format(fov * 1000, Nx, Ny, TE * 1000,
                                                                                               TR * 1000, flip * 180 / pi))
    print('GRE sequence constructed')
    return seq


def make_pulseq_irse(fov,n,thk,fa,tr,te,ti,write=True):
    kwargs_for_opts = {"max_grad": 33, "grad_unit": "mT/m", "max_slew": 100, "slew_unit": "T/m/s",
                       "rf_dead_time": 10e-6,
                       "adc_dead_time": 10e-6}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)

    Nx = n
    Ny = n
    TI,TE,TR = ti,te,tr
    flip = fa * pi / 180
    kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 2e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)

    delta_k = 1 / fov
    kWidth = Nx * delta_k
    readoutTime = system.grad_raster_time * Nx
    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": kWidth, "flat_time": readoutTime}
    gx = make_trapezoid(kwargs_for_gx)
    kwargs_for_adc = {"num_samples": Nx, "system": system, "duration": gx.flat_time, "delay": gx.rise_time}
    adc = makeadc(kwargs_for_adc)

    kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gx.area/2, # TODO
                        "duration": gx.rise_time + gx.fall_time + readoutTime / 2}


    gx_pre = make_trapezoid(kwargs_for_gxpre)
    kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": 2e-3}
    gz_reph = make_trapezoid(kwargs_for_gz_reph)

    flip = 180 * pi / 180
    kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 2e-3, "slice_thickness": thk,
                       "apodization": 0.5, "time_bw_product": 4}
    rf180, gz180 = make_sinc_pulse(kwargs_for_sinc, 2)

    delayTE1 = TE / 2 - max(calc_duration(gz_reph), calc_duration(gx_pre)) - calc_duration(rf) / 2 - calc_duration(
        rf180) / 2
    delayTE2 = TE / 2 - calc_duration(gx) / 2 - calc_duration(rf180) / 2
    delayTE3 = TR - TE - calc_duration(rf) / 2 - calc_duration(gx) / 2
    delay1 = make_delay(delayTE1)
    delay2 = make_delay(delayTE2)
    delay3 = make_delay(delayTE3)

    for inv in range(len(TI)):
        for i in range(Ny):
            # Inversion Recovery part
            seq.add_block(rf180)# Non-selective at the moment, could be extended to make this selective/adiabatic
            seq.add_block(make_delay(TI[inv] - calc_duration(rf) / 2 - calc_duration(rf180) / 2))  # Inversion time delay
            # Spin echo part
            seq.add_block(rf, gz)  # 90-deg pulse

            kwargs_for_gy_pre = {"channel": 'y', "system": system, "area": -(Ny / 2 - i) * delta_k,
                                 "duration": gx.rise_time + gx.fall_time + readoutTime / 2}

            gy_pre = make_trapezoid(kwargs_for_gy_pre)  # Phase encoding gradient
            seq.add_block(gx_pre, gy_pre, gz_reph)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
            seq.add_block(delay1)  # Delay 1: until 180-deg pulse
            seq.add_block(rf180, gz180)  # 180 deg pulse for SE
            seq.add_block(delay2)  # Delay 2: until readout
            seq.add_block(gx, adc)  # Readout!
            seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        if len(TI) == 1:
            seq.write("irse_fov{:.0f}mm_Nx{:d}_Ny{:d}_TI{:.0f}ms_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nx, Ny, TI[0] * 1000, TE * 1000, TR * 1000))
        else:
            seq.write("irse_fov{:.0f}mm_Nx{:d}_Ny{:d}_multiTI_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nx, Ny, TE * 1000, TR * 1000))

    print('IRSE sequence constructed')
    return seq

