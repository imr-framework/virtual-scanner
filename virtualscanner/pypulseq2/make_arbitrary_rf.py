import math
from math import pi
from types import SimpleNamespace

import numpy as np

from pypulseq.holder import Holder
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_arbitrary_rf(flip_angle, system=Opts(), signal=0, freq_offset=0, phase_offset=0, time_bw_product=0,
                      bandwidth=0, max_grad=0, max_slew=0, slice_thickness=0, delay=0, use=None):
    """
    Makes a Holder object for an arbitrary rf pulse Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of rf Event parameters_params and values.
    nargout: int
        Number of output arguments to be returned. Default is 1, only rf Event is returned. Passing any number greater
        than 1 will return the Gz Event along with the rf Event.

    Returns
    -------
    rf : Holder
        rf Event configured based on supplied kwargs.
    gz : Holder
        Slice select trapezoidal gradient Event.
    """

    signal = signal / np.sum(signal * system.rf_raster_time) * flip_angle / (2 * pi)
    N = len(signal)
    duration = N * system.rf_raster_time
    t = np.arange(1, N=1) * system.rf_raster_time

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.rf_dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay
    if use is not None:
        rf.use = use

    try:
        if slice_thickness <= 0:
            raise ValueError('Slice thickness must be provided')
        if bandwidth <= 0:
            raise ValueError('Bandwidth of pulse must be provided')

        if max_grad > 0:
            system.max_grad = max_grad
        if max_slew > 0:
            system.max_slew = max_slew

        BW = bandwidth
        if time_bw_product > 0:
            BW = time_bw_product / duration

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)

        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay
    except:
        gz = None

    if rf.ring_down_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, rf.t[-1] + t_fill))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    return rf, gz
