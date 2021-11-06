import numpy as np


GAMMA_BAR = 42.5775e6

def find_precessing_time(blk,dt):
    """Helper function that finds and returns longest duration among Gx, Gy, and Gz for use in SpinGroup.fpwg()

    Parameters
    ----------
    blk : dict
        Pulseq Block obtained from seq.get_block()
    dt : float
        Gradient raster time for calculating duration of only arbitrary gradients ('grad' instead of 'trap')

    Returns
    -------
    max_time : float
        Maximum gradient time, in seconds, among the three gradients Gx, Gy, and Gz

    """
    grad_times = []
    for g_name in ['gx','gy','gz']:
        #if blk.__contains__(g_name):
        if hasattr(blk, g_name):
            #g = blk[g_name]
            g = blk.__getattribute__(g_name)
            tg = (g.rise_time + g.flat_time + g.fall_time) if g.type == 'trap' else len(np.squeeze(g.t))*dt
            grad_times.append(tg)
    return max(grad_times)


def combine_gradients(blk,dt=0,timing=(),delay=0):
    """Helper function that merges multiple gradients into a format for simulation

    Interpolate x, y, and z gradients starting from time 0
    at dt intervals, for as long as the longest gradient lasts
    and combine them into a 3 x N array

    Parameters
    ----------
    blk : dict
        Pulseq block obtained from seq.get_block()
    dt : float, optional
        Raster time used in interpolating gradients, in seconds
        Default is 0 - in this case, timing is supposed to be inputted
    timing : numpy.ndarray, optional
        Time points at which gradients are interpolated, in seconds
        Default is () - in this case, dt is supposed to be inputted
    delay : float, optional
            Adds an additional time interval in seconds at the beginning of the interpolation
            Default is 0; when nonzero it is only used in ADC sampling to realize ADC delay

    Returns
    -------
    grad : numpy.ndarray
        Gradient shape in Tesla/meter
    grad_timing : numpy.ndarray
        Gradient timing in seconds
    duration: float
        Duration of input block in seconds
    grad_type : str
        Type of gradient

    Notes
    -----
    Only input one argument between dt and timing and not the other

    """
    grad_timing = []
    duration = 0
    if dt != 0:
        duration = find_precessing_time(blk,dt)
        grad_timing = np.concatenate(([0],np.arange(delay,duration+dt,dt)))
    elif len(timing) != 0:
        duration = timing[-1] - timing[0]
        grad_timing = timing

    grad = []
    g = None
    # Interpolate gradient values at desired time points
    for g_name in ['gx','gy','gz']:
       # if blk.__contains__(g_name):
        if hasattr(blk, g_name):
            #g = blk[g_name]
            g = blk.__getattribute__(g_name)
            g_time, g_shape = ([0, g.rise_time, g.rise_time + g.flat_time, g.rise_time + g.flat_time + g.fall_time],
                               [0,g.amplitude/GAMMA_BAR,g.amplitude/GAMMA_BAR,0]) if g.type == 'trap'\
                               else (g.t, g.waveform/GAMMA_BAR)
            g_time = np.array(g_time)
            grad.append(np.interp(x=grad_timing,xp=g_time,fp=g_shape))
        else:
            grad.append(np.zeros(np.shape(grad_timing)))
    if g is not None:
        grad_type = g.type
    else:
        grad_type = None
    return np.array(grad), grad_timing, duration, grad_type
