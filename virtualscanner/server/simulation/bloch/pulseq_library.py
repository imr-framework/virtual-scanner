# Copyright of the Board of Trustees of Columbia University in the City of New York
"""
Library for generating pulseq sequences: GRE, SE, IRSE, EPI
"""


import copy
from math import pi, sqrt, ceil, floor

import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

GAMMA_BAR = 42.5775e6
GAMMA = 2*pi*GAMMA_BAR


def make_pulseq_gre(fov,n,thk,fa,tr,te,enc='xyz',slice_locs=None,write=False):
    """Makes a gradient-echo sequence

    2D orthogonal multi-slice gradient-echo pulse sequence with Cartesian encoding
    Orthogonal means that each of slice-selection, phase encoding, and frequency encoding
    aligns with the x, y, or z directions

    Parameters
    ----------
    fov : float
        Field-of-view in meters (isotropic)
    n : int
        Matrix size (isotropic)
    thk : float
        Slice thickness in meters
    fa : float
        Flip angle in degrees
    tr : float
        Repetition time in seconds
    te : float
        Echo time in seconds
    enc : str, optional
        Spatial encoding directions
        1st - readout; 2nd - phase encoding; 3rd - slice select
        Default 'xyz' means axial(z) slice with readout in x and phase encoding in y
    slice_locs : array_like, optional
        Slice locations from isocenter in meters
        Default is None which means a single slice at the center
    write : bool, optional
        Whether to write seq into file; default is False

    Returns
    -------
    seq : Sequence
        Pulse sequence as a Pulseq object

    """
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)


    Nf = n
    Np = n
    flip = fa * pi / 180
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip, system=system, duration=4e-3, slice_thickness=thk,
                               apodization=0.5, time_bw_product=4)

    g_ss.channel = enc[2]

    delta_k = 1 / fov
    kWidth = Nf * delta_k

    # Readout and ADC
    readoutTime = 6.4e-3
    g_ro= make_trapezoid(channel=enc[0],system=system,flat_area=kWidth,flat_time=readoutTime)
    adc = make_adc(num_samples=Nf, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # Readout rewinder
    g_ro_pre = make_trapezoid(channel=enc[0],system=system,area=-g_ro.area/2,duration=2e-3)
    # Slice refocusing
    g_ss_reph = make_trapezoid(channel=enc[2],system=system,area=-g_ss.area/2,duration=2e-3)
    phase_areas = (np.arange(Np) - (Np / 2)) * delta_k

    # TE, TR = 10e-3, 1000e-3
    TE, TR = te,tr
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
            g_pe = make_trapezoid(channel=enc[1],system=system,area=phase_areas[i],duration=2e-3)
            seq.add_block(g_ro_pre, g_pe, g_ss_reph)
            seq.add_block(delay1)
            seq.add_block(g_ro,adc)
            seq.add_block(delay2)

    if write:
        seq.write("gre_fov{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:.0f}deg.seq".format(fov * 1000, Nf, Np, TE * 1000,
                                                                                               TR * 1000, flip * 180 / pi))
    print('GRE sequence constructed')
    return seq


def make_pulseq_gre_oblique(fov,n,thk,fa,tr,te,enc='xyz',slice_locs=None,write=False):
    """Makes a gradient-echo sequence in any plane

        2D oblique multi-slice gradient-echo pulse sequence with Cartesian encoding
        Oblique means that each of slice-selection, phase encoding, and frequency encoding
        can point in any specified direction

        Parameters
        ----------
        fov : array_like
            Isotropic field-of-view, or length-2 list [fov_readout, fov_phase], in meters
        n : array_like
            Isotropic matrix size, or length-2 list [n_readout, n_phase]
        thk : float
            Slice thickness in meters
        fa : float
            Flip angle in degrees
        tr : float
            Repetition time in seconds
        te : float
            Echo time in seconds
        enc : str or array_like, optional
            Spatial encoding directions
            1st - readout; 2nd - phase encoding; 3rd - slice select
            - Use str with any permutation of x, y, and z to obtain orthogonal slices
            e.g. The default 'xyz' means axial(z) slice with readout in x and phase encoding in y
            - Use list to indicate vectors in the encoding directions for oblique slices
            They should be perpendicular to each other, but not necessarily unit vectors
            e.g. [(2,1,0),(-1,2,0),(0,0,1)] rotates the two in-plane encoding directions for an axial slice
        slice_locs : array_like, optional
            Slice locations from isocenter in meters
            Default is None which means a single slice at the center
        write : bool, optional
            Whether to write seq into file; default is False

        Returns
        -------
        seq : Sequence
            Pulse sequence as a Pulseq object

        """

    # System options
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)

    # Calculate unit gradients for ss, fe, pe
    ug_fe, ug_pe, ug_ss = parse_enc(enc)

    # Sequence parameters
    Nf, Np = (n,n) if isinstance(n,int) else (n[0], n[1])
    delta_k_ro, delta_k_pe = (1/fov,1/fov) if isinstance(fov,float) else (1/fov[0], 1/fov[1])
    kWidth_ro = Nf * delta_k_ro
    flip = fa * pi / 180

    # Slice select: RF and gradient
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip,system=system,duration=4e-3,slice_thickness=thk,
                               apodization=0.5, time_bw_product=4, return_gz=True)
    g_ss_x, g_ss_y, g_ss_z = make_oblique_gradients(g_ss,ug_ss)

    # Readout and ADC
#    readoutTime = 6.4e-3
    dwell = 10e-6
    g_ro= make_trapezoid(channel='x',system=system,flat_area=kWidth_ro, flat_time=dwell*Nf)
    g_ro_x, g_ro_y, g_ro_z = make_oblique_gradients(g_ro,ug_fe)#
    adc = make_adc(num_samples=Nf, duration=g_ro.flat_time,delay=g_ro.rise_time)

    # Readout rewinder
    g_ro_pre = make_trapezoid(channel='x',system=system,area=-g_ro.area/2,duration=2e-3)
    g_ro_pre_x, g_ro_pre_y, g_ro_pre_z = make_oblique_gradients(g_ro_pre,ug_fe)#

    # Slice refocusing
    g_ss_reph = make_trapezoid(channel='z',system=system,area=-g_ss.area/2,duration=2e-3)
    g_ss_reph_x, g_ss_reph_y, g_ss_reph_z = make_oblique_gradients(g_ss_reph, ug_ss)

    # Prepare phase areas
    phase_areas = (np.arange(Np) - (Np / 2)) * delta_k_pe

    TE, TR = te,tr
    delayTE = TE - calc_duration(g_ro_pre) - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2
    delayTR = TR - calc_duration(g_ro_pre) - calc_duration(g_ss) - calc_duration(g_ro) - delayTE
    delay1 = make_delay(delayTE)
    delay2 = make_delay(delayTR)

    if slice_locs is None:
        locs = [0]
    else:
        locs = slice_locs

    # Construct sequence!
    for u in range(len(locs)):
        # add frequency offset
        rf.freq_offset = g_ss.amplitude * locs[u]
        for i in range(Np):
            seq.add_block(rf,g_ss_x, g_ss_y, g_ss_z)
            g_pe = make_trapezoid(channel='y',system=system,area=phase_areas[i],duration=2e-3)

            g_pe_x, g_pe_y, g_pe_z = make_oblique_gradients(g_pe,ug_pe)

            pre_grads_list = [g_ro_pre_x, g_ro_pre_y, g_ro_pre_z,
                             g_ss_reph_x, g_ss_reph_y, g_ss_reph_z,
                             g_pe_x, g_pe_y, g_pe_z]

            gtx, gty, gtz = combine_trap_grad_xyz(gradients=pre_grads_list,system=system, dur=2e-3)

            seq.add_block(gtx, gty, gtz)
            seq.add_block(delay1)
            seq.add_block(g_ro_x, g_ro_y, g_ro_z, adc)
            seq.add_block(delay2)

    if write:
        seq.write("gre_fov{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:.0f}deg.seq".format(fov * 1000, Nf, Np, TE * 1000,
                                                                                               TR * 1000, flip * 180 / pi))
    print('GRE sequence constructed')
    return seq


def make_pulseq_irse(fov,n,thk,fa,tr,te,ti,enc='xyz',slice_locs=None,write=False):
    """Makes an Inversion Recovery Spin Echo (IRSE) sequence

        2D orthogonal multi-slice IRSE pulse sequence with Cartesian encoding
        Orthogonal means that each of slice-selection, phase encoding, and frequency encoding
        aligns with the x, y, or z directions

        Parameters
        ----------
        fov : float
            Field-of-view in meters (isotropic)
        n : int
            Matrix size (isotropic)
        thk : float
            Slice thickness in meters
        fa : float
            Flip angle in degrees
        tr : float
            Repetition time in seconds
        te : float
            Echo time in seconds
        ti : float
            Inversion time in seconds
        enc : str, optional
            Spatial encoding directions
            1st - readout; 2nd - phase encoding; 3rd - slice select
            Default 'xyz' means axial(z) slice with readout in x and phase encoding in y
        slice_locs : array_like, optional
            Slice locations from isocenter in meters
            Default is None which means a single slice at the center
        write : bool, optional
            Whether to write seq into file; default is False

        Returns
        -------
        seq : Sequence
            Pulse sequence as a Pulseq object

        """
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)

    # Parameters
    Nf = n
    Np = n
    delta_k = 1 / fov
    kWidth = Nf * delta_k

    TI,TE,TR = ti,te,tr

    if np.shape(TI) == ():
        TI = [TI]


    # Non-180 pulse
    flip1 = fa * pi / 180
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip1, system=system, duration=2e-3, slice_thickness=thk,
                               apodization=0.5, time_bw_product=4)

    g_ss.channel = enc[2]

    # 180 pulse
    flip2 = 180 * pi / 180
    rf180, g_ss180, __ = make_sinc_pulse(flip_angle=flip2, system=system,duration=2e-3, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=4)
    g_ss180.channel = enc[2]

    # Readout gradient & ADC
    readoutTime = 6.4e-3

    g_ro = make_trapezoid(channel=enc[0],system=system, flat_area=kWidth, flat_time=readoutTime)
    adc = make_adc(num_samples=Nf, system=system, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # RO rewinder gradient
    g_ro_pre = make_trapezoid(channel=enc[0],system=system,area=g_ro.area/2,duration=2e-3)

    # Slice refocusing gradient
    g_ss_reph = make_trapezoid(channel=enc[2],system=system,area=-g_ss.area/2,duration=2e-3)

    # Delays

    delayTE1 = TE / 2 - max(calc_duration(g_ss_reph), calc_duration(g_ro_pre)) - calc_duration(g_ss) / 2 - calc_duration(
        g_ss180) / 2
    delayTE2 = TE / 2 - calc_duration(g_ro) / 2 - calc_duration(g_ss180) / 2
    delayTE3 = TR - TE - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2

    print('dur rf', calc_duration(rf),'dur gss:' ,calc_duration(g_ss))

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
                seq.add_block(rf180, g_ss180)# Selective; potentially extended to be non-selective or adiabatic
                seq.add_block(make_delay(TI[inv] - calc_duration(rf) / 2 - calc_duration(rf180) / 2))  # Inversion time delay
                # Spin echo part
                seq.add_block(rf, g_ss)  # 90-deg pulse
                g_pe_pre = make_trapezoid(channel=enc[1],system=system,area=-(Np/2-i)*delta_k,
                                          duration=2e-3)  # Phase encoding gradient
                seq.add_block(g_ro_pre, g_pe_pre, g_ss_reph)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
                seq.add_block(delay1)  # Delay 1: until 180-deg pulse
                seq.add_block(rf180, g_ss180)  # 180 deg pulse for SE
                seq.add_block(delay2)  # Delay 2: until readout
                seq.add_block(g_ro, adc)  # Readout!
                seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        if len(TI) == 1:
            seq.write("irse_fov{:.0f}mm_Nf{:d}_Np{:d}_TI{:.0f}ms_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nf, Np, TI[0] * 1000, TE * 1000, TR * 1000))
        else:
            seq.write("irse_fov{:.0f}mm_Nf{:d}_Np{:d}_multiTI_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nf, Np, TE * 1000, TR * 1000))

    print('IRSE sequence constructed')
    return seq

def make_pulseq_irse_oblique(fov,n,thk,fa,tr,te,ti,enc='xyz',slice_locs=None,write=False):
    """Makes an Inversion Recovery Spin Echo (IRSE) sequence in any plane

        2D oblique multi-slice IRSE pulse sequence with Cartesian encoding
        Oblique means that each of slice-selection, phase encoding, and frequency encoding
        can point in any specified direction

        Parameters
        ----------
        fov : array_like
            Isotropic field-of-view, or length-2 list [fov_readout, fov_phase], in meters
        n : array_like
            Isotropic matrix size, or length-2 list [n_readout, n_phase]
        thk : float
            Slice thickness in meters
        fa : float
            Flip angle in degrees
        tr : float
            Repetition time in seconds
        te : float
            Echo time in seconds
        ti : float
            Inversion time in seconds
        enc : str or array_like, optional
            Spatial encoding directions
            1st - readout; 2nd - phase encoding; 3rd - slice select
            - Use str with any permutation of x, y, and z to obtain orthogonal slices
            e.g. The default 'xyz' means axial(z) slice with readout in x and phase encoding in y
            - Use list to indicate vectors in the encoding directions for oblique slices
            They should be perpendicular to each other, but not necessarily unit vectors
            e.g. [(2,1,0),(-1,2,0),(0,0,1)] rotates the two in-plane encoding directions for an axial slice
        slice_locs : array_like, optional
            Slice locations from isocenter in meters
            Default is None which means a single slice at the center
        write : bool, optional
            Whether to write seq into file; default is False


        Returns
        -------
        seq : Sequence
            Pulse sequence as a Pulseq object

        """
    # System options
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)

    # Sequence parameters
    ug_fe, ug_pe, ug_ss = parse_enc(enc)
    Nf, Np = (n,n) if isinstance(n,int) else (n[0], n[1])
    delta_k_ro, delta_k_pe = (1/fov,1/fov) if isinstance(fov,float) else (1/fov[0], 1/fov[1])
    kWidth_ro = Nf * delta_k_ro
    TI,TE,TR = ti,te,tr

    if np.shape(TI) == ():
        TI = [TI]

    # Non-180 pulse
    flip1 = fa * pi / 180
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip1, system=system, duration=2e-3, slice_thickness=thk,
                                apodization=0.5, time_bw_product=4)
    g_ss_x, g_ss_y, g_ss_z = make_oblique_gradients(g_ss, ug_ss)

    # 180 pulse
    flip2 = 180 * pi / 180
    rf180, g_ss180, __ = make_sinc_pulse(flip_angle=flip2, system=system, duration=2e-3, slice_thickness=thk,
                                    apodization=0.5, time_bw_product=4)
    g_ss180_x, g_ss180_y, g_ss180_z = make_oblique_gradients(g_ss180, ug_ss)

    # Readout gradient & ADC
    readoutTime = 6.4e-3

    g_ro = make_trapezoid(channel='x', system=system, flat_area=kWidth_ro, flat_time= readoutTime)
    g_ro_x, g_ro_y, g_ro_z = make_oblique_gradients(g_ro, ug_fe)

    adc = make_adc(num_samples=Nf, system=system, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # RO rewinder gradient
    g_ro_pre = make_trapezoid(channel=enc[0], system=system, area=g_ro.area/2,duration=2e-3)
    g_ro_pre_x, g_ro_pre_y, g_ro_pre_z = make_oblique_gradients(g_ro_pre,ug_fe)#

    # Slice refocusing gradient
    g_ss_reph = make_trapezoid(channel=enc[2],system=system,area=-g_ss.area/2,duration=2e-3)
    g_ss_reph_x, g_ss_reph_y, g_ss_reph_z = make_oblique_gradients(g_ss_reph, ug_ss)

    # Delays
    delayTE1 = TE / 2 - max(calc_duration(g_ss_reph), calc_duration(g_ro_pre))\
               - calc_duration(g_ss) / 2 - calc_duration(g_ss180) / 2
    delayTE2 = TE / 2 - calc_duration(g_ro) / 2 - calc_duration(g_ss180) / 2
    delayTE3 = TR - TE - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2

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
                seq.add_block(rf180, g_ss180_x, g_ss180_y, g_ss180_z)# Non-selective at the moment; could be extended to make this selective/adiabatic
                seq.add_block(make_delay(TI[inv] - calc_duration(rf) / 2 - calc_duration(rf180) / 2))  # Inversion time delay
                # Spin echo part
                seq.add_block(rf, g_ss_x, g_ss_y, g_ss_z)  # 90-deg pulse
                g_pe = make_trapezoid(channel='y', system=system, area=-(Np /2 - i)*delta_k_pe, duration=2e-3)  # Phase encoding gradient
                g_pe_x, g_pe_y, g_pe_z = make_oblique_gradients(g_pe, ug_pe)

                pre_grads_list = [g_ro_pre_x, g_ro_pre_y, g_ro_pre_z,
                                  g_ss_reph_x, g_ss_reph_y, g_ss_reph_z,
                                  g_pe_x, g_pe_y, g_pe_z]
                gtx, gty, gtz = combine_trap_grad_xyz(pre_grads_list,system,2e-3)

                seq.add_block(gtx, gty, gtz)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
                seq.add_block(delay1)  # Delay 1: until 180-deg pulse
                seq.add_block(rf180, g_ss180_x, g_ss180_y, g_ss180_z)  # 180 deg pulse for SE
                seq.add_block(delay2)  # Delay 2: until readout
                seq.add_block(g_ro_x, g_ro_y, g_ro_z, adc)  # Readout!
                seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        if len(TI) == 1:
            seq.write("irse_fov{:.0f}mm_Nf{:d}_Np{:d}_TI{:.0f}ms_TE{:.0f}ms_TR{:.0f}ms_FA{:d}deg.seq".format(fov * 1000,
                                                                            Nf, Np, TI[0] * 1000, TE * 1000, TR * 1000, fa))
        else:
            seq.write("irse_fov{:.0f}mm_Nf{:d}_Np{:d}_multiTI_TE{:.0f}ms_TR{:.0f}ms_FA{:d}deg.seq".format(fov * 1000,
                                                                            Nf, Np, TE * 1000, TR * 1000, fa))

    print('IRSE (oblique) sequence constructed')
    return seq

def make_pulseq_se(fov,n,thk,fa,tr,te,enc='xyz',slice_locs=None,write=False):
    """Makes a Spin Echo (SE) sequence

    2D orthogonal multi-slice Spin-Echo pulse sequence with Cartesian encoding
    Orthogonal means that each of slice-selection, phase encoding, and frequency encoding
    aligns with the x, y, or z directions

    Parameters
    ----------
    fov : float
        Field-of-view in meters (isotropic)
    n : int
        Matrix size (isotropic)
    thk : float
        Slice thickness in meters
    fa : float
        Flip angle in degrees
    tr : float
        Repetition time in seconds
    te : float
        Echo time in seconds
    enc : str, optional
        Spatial encoding directions
        1st - readout; 2nd - phase encoding; 3rd - slice select
        Default 'xyz' means axial(z) slice with readout in x and phase encoding in y
    slice_locs : array_like, optional
        Slice locations from isocenter in meters
        Default is None which means a single slice at the center
    write : bool, optional
        Whether to write seq into file; default is False

    Returns
    -------
    seq : Sequence
        Pulse sequence as a Pulseq object

    """
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)

    # Parameters
    Nf = n
    Np = n
    delta_k = 1 / fov
    kWidth = Nf * delta_k

    TE,TR = te,tr


    # Non-180 pulse
    flip1 = fa * pi / 180
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip1, system=system, duration=2e-3, slice_thickness=thk,
                                apodization=0.5, time_bw_product=4)
    g_ss.channel = enc[2]


    # 180 pulse
    flip2 = 180 * pi / 180
    rf180, g_ss180, __ = make_sinc_pulse(flip_angle=flip2, system=system, duration=2e-3, slice_thickness=thk,
                                    apodization=0.5, time_bw_product=4)
    g_ss180.channel = enc[2]

    # Readout gradient & ADC
#    readoutTime = system.grad_raster_time * Nf
    readoutTime = 6.4e-3
    g_ro = make_trapezoid(channel=enc[0],system=system,flat_area=kWidth,flat_time=readoutTime)
    adc = make_adc(num_samples=Nf, system=system, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # RO rewinder gradient
    g_ro_pre = make_trapezoid(channel=enc[0],system=system,area=g_ro.area/2,duration=2e-3)

    # Slice refocusing gradient
    g_ss_reph = make_trapezoid(channel=enc[2],system=system,area=-g_ss.area/2,duration=2e-3)

    # Delays
    delayTE1 = (TE - 2*max(calc_duration(g_ss_reph), calc_duration(g_ro_pre)) - calc_duration(g_ss) - calc_duration(
        g_ss180))/2
  # delayTE2 = TE / 2 - calc_duration(g_ro) / 2 - calc_duration(g_ss180) / 2
    delayTE2 = (TE - calc_duration(g_ro) - calc_duration(g_ss180))/2
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
            g_pe_pre = make_trapezoid(channel=enc[1],system=system,area=-(Np/2-i)*delta_k,duration=2e-3)  # Phase encoding gradient
            seq.add_block(g_ro_pre, g_pe_pre, g_ss_reph)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
            seq.add_block(delay1)  # Delay 1: until 180-deg pulse
            seq.add_block(rf180, g_ss180)  # 180 deg pulse for SE
            seq.add_block(delay2)  # Delay 2: until readout
            seq.add_block(g_ro, adc)  # Readout!
            seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        seq.write("se_fov{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms.seq".format(fov * 1000, Nf, Np, TE * 1000, TR * 1000))


    print('Spin echo sequence constructed')
    return seq

def make_pulseq_se_oblique(fov,n,thk,fa,tr,te,enc='xyz',slice_locs=None,write=False):
    """Makes a Spin Echo (SE) sequence in any plane

        2D oblique multi-slice Spin-Echo pulse sequence with Cartesian encoding
        Oblique means that each of slice-selection, phase encoding, and frequency encoding
        can point in any specified direction

        Parameters
        ----------
        fov : array_like
            Isotropic field-of-view, or length-2 list [fov_readout, fov_phase], in meters
        n : array_like
            Isotropic matrix size, or length-2 list [n_readout, n_phase]
        thk : float
            Slice thickness in meters
        fa : float
            Flip angle in degrees
        tr : float
            Repetition time in seconds
        te : float
            Echo time in seconds
        enc : str or array_like, optional
            Spatial encoding directions
            1st - readout; 2nd - phase encoding; 3rd - slice select
            - Use str with any permutation of x, y, and z to obtain orthogonal slices
            e.g. The default 'xyz' means axial(z) slice with readout in x and phase encoding in y
            - Use list to indicate vectors in the encoding directions for oblique slices
            They should be perpendicular to each other, but not necessarily unit vectors
            e.g. [(2,1,0),(-1,2,0),(0,0,1)] rotates the two in-plane encoding directions for an axial slice
        slice_locs : array_like, optional
            Slice locations from isocenter in meters
            Default is None which means a single slice at the center
        write : bool, optional
            Whether to write seq into file; default is False

        Returns
        -------
        seq : Sequence
            Pulse sequence as a Pulseq object

        """

    # System options
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)

    # Sequence parameters
    ug_fe, ug_pe, ug_ss = parse_enc(enc)
    Nf, Np = (n,n) if isinstance(n,int) else (n[0], n[1])
    delta_k_ro, delta_k_pe = (1/fov,1/fov) if isinstance(fov,float) else (1/fov[0], 1/fov[1])
    kWidth_ro = Nf * delta_k_ro
    TE,TR = te,tr

    # Non-180 pulse
    flip1 = fa * pi / 180
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip1, system=system, duration=2e-3, slice_thickness=thk,
                                apodization=0.5, time_bw_product=4, return_gz=True)
    g_ss_x, g_ss_y, g_ss_z = make_oblique_gradients(g_ss,ug_ss)

    # 180 pulse
    flip2 = 180 * pi / 180
    rf180, g_ss180, __ = make_sinc_pulse(flip_angle=flip2, system=system, duration=2e-3, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=4, return_gz=True)
    g_ss180_x, g_ss180_y, g_ss180_z = make_oblique_gradients(g_ss180,ug_ss)

    # Readout gradient & ADC
    readoutTime = 6.4e-3
    g_ro = make_trapezoid(channel='x',system=system,flat_area=kWidth_ro, flat_time=readoutTime)
    g_ro_x, g_ro_y, g_ro_z = make_oblique_gradients(g_ro, ug_fe)
    adc = make_adc(num_samples=Nf, system=system, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # RO rewinder gradient
    g_ro_pre = make_trapezoid(channel='x',system=system,area=g_ro.area/2,duration=2e-3)
    g_ro_pre_x, g_ro_pre_y, g_ro_pre_z = make_oblique_gradients(g_ro_pre, ug_fe)

    # Slice refocusing gradient
    g_ss_reph = make_trapezoid(channel='z',system=system,area=-g_ss.area/2,duration=2e-3)
    g_ss_reph_x, g_ss_reph_y, g_ss_reph_z = make_oblique_gradients(g_ss_reph, ug_ss)

    # Delays
    delayTE1 = (TE - 2*max(calc_duration(g_ss_reph), calc_duration(g_ro_pre)) - calc_duration(g_ss) - calc_duration(
        g_ss180))/2
    delayTE2 = (TE - calc_duration(g_ro) - calc_duration(g_ss180))/2
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
            seq.add_block(rf, g_ss_x, g_ss_y, g_ss_z)  # 90-deg pulse
            g_pe = make_trapezoid(channel='y',system=system,area=-(Np/2 - i)*delta_k_pe, duration=2e-3)  # Phase encoding gradient
            g_pe_x, g_pe_y, g_pe_z = make_oblique_gradients(g_pe, ug_pe)

            pre_grads_list = [g_ro_pre_x, g_ro_pre_y, g_ro_pre_z,
                              g_ss_reph_x, g_ss_reph_y, g_ss_reph_z,
                              g_pe_x, g_pe_y, g_pe_z]
            gtx, gty, gtz = combine_trap_grad_xyz(pre_grads_list, system, 2e-3)


            seq.add_block(gtx,gty,gtz)  # Add a combination of ro rewinder, phase encoding, and slice refocusing
            seq.add_block(delay1)  # Delay 1: until 180-deg pulse
            seq.add_block(rf180, g_ss180_x, g_ss180_y, g_ss180_z)  # 180 deg pulse for SE
            seq.add_block(delay2)  # Delay 2: until readout
            seq.add_block(g_ro_x, g_ro_y, g_ro_z, adc)  # Readout!
            seq.add_block(delay3)  # Delay 3: until next inversion pulse

    if write:
        seq.write("se_fov{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:d}deg.seq".format(fov * 1000, Nf, Np, TE * 1000, TR * 1000, fa))


    print('Spin echo sequence (oblique) constructed')
    return seq


# TODO multi-shot epi needs to be tested on scanner! : )
def make_pulseq_epi_oblique(fov,n,thk,fa,tr,te,enc='xyz',slice_locs=None,echo_type="se",n_shots=1,seg_type='blocked',write=False):
    """Makes an Echo Planar Imaging (EPI) sequence in any plane

        2D oblique multi-slice EPI pulse sequence with Cartesian encoding
        Oblique means that each of slice-selection, phase encoding, and frequency encoding
        can point in any specified direction

        Parameters
        ----------
        fov : array_like
            Isotropic field-of-view, or length-2 list [fov_readout, fov_phase], in meters
        n : array_like
            Isotropic matrix size, or length-2 list [n_readout, n_phase]
        thk : float
            Slice thickness in meters
        fa : float
            Flip angle in degrees
        tr : float
            Repetition time in seconds
        te : float
            Echo time in seconds
        enc : str or array_like, optional
            Spatial encoding directions
            1st - readout; 2nd - phase encoding; 3rd - slice select
            - Use str with any permutation of x, y, and z to obtain orthogonal slices
            e.g. The default 'xyz' means axial(z) slice with readout in x and phase encoding in y
            - Use list to indicate vectors in the encoding directions for oblique slices
            They should be perpendicular to each other, but not necessarily unit vectors
            e.g. [(2,1,0),(-1,2,0),(0,0,1)] rotates the two in-plane encoding directions for an axial slice
        slice_locs : array_like, optional
            Slice locations from isocenter in meters
            Default is None which means a single slice at the center
        echo_type : str, optional {'se','gre'}
            Type of echo generated
            se (default) - spin echo (an 180 deg pulse is used)
            gre - gradient echo
        n_shots : int, optional
            Number of shots used to encode each slicel; default is 1
        seg_type : str, optional {'blocked','interleaved'}
            Method to divide up k-space in the case of n_shots > 1; default is 'blocked'
            'blocked' - each shot covers a rectangle, with no overlap between shots
            'interleaved' - each shot samples the full k-space but with wider phase steps

        write : bool, optional
            Whether to write seq into file; default is False

        Returns
        -------
        seq : Sequence
            Pulse sequence as a Pulseq object
        ro_dirs : numpy.ndarray
            List of 0s and 1s indicating direction of readout
            0 - left to right
            1 - right to left (needs to be reversed at recon)
        ro_order : numpy.ndarray
            Order in which to re-arrange the readout lines
            It is [] for blocked acquisition (retain original order)

        """
    # Multi-slice, multi-shot (>=1)
    # TE is set to be where the trajectory crosses the center of k-space

    # System options
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)


    ug_fe, ug_pe, ug_ss = parse_enc(enc)

    # Sequence parameters
    Nf, Np = (n,n) if isinstance(n,int) else (n[0], n[1])
    delta_k_ro, delta_k_pe = (1/fov,1/fov) if isinstance(fov,float) else (1/fov[0], 1/fov[1])
    kWidth_ro = Nf * delta_k_ro
    TE,TR = te,tr
    flip = fa * pi / 180

    # RF Pulse (first)
    rf, g_ss, __ = make_sinc_pulse(flip_angle=flip, system=system,duration=2.5e-3, slice_thickness=thk,
                               apodization=0.5, time_bw_product=4,return_gz=True)
    g_ss_x, g_ss_y, g_ss_z = make_oblique_gradients(g_ss, ug_ss)


    # Readout gradients
#    readoutTime = Nf * 4e-6
    dwell=1e-5
    readoutTime = Nf*dwell
    g_ro_pos = make_trapezoid(channel='x',system=system,flat_area=kWidth_ro,flat_time=readoutTime)
    g_ro_pos_x, g_ro_pos_y, g_ro_pos_z = make_oblique_gradients(g_ro_pos,ug_fe)
    g_ro_neg = copy.deepcopy(g_ro_pos)
    modify_gradient(g_ro_neg,scale=-1)
    g_ro_neg_x, g_ro_neg_y, g_ro_neg_z = make_oblique_gradients(g_ro_neg,ug_fe)

    # TODO make sure delay is a multiple of gradient raster time
#    adc = make_adc(num_samples=Nf, system=system, duration=g_ro_pos.flat_time, delay=g_ro_pos.rise_time+dwell/2)
    adc = make_adc(num_samples=Nf, system=system, duration=g_ro_pos.flat_time, delay=g_ro_pos.rise_time)
    print("ADC delay: ", adc.delay)

    pre_time = 8e-4

    # 180 deg pulse for SE
    if echo_type == "se":
        # RF Pulse (180 deg for SE)
        flip180 = 180 * pi / 180
        rf180, g_ss180, __ = make_sinc_pulse(flip_angle=flip180, system=system,duration=2.5e-3,slice_thickness=thk,
                                         apodization=0.5, time_bw_product=4, return_gz=True)
        g_ss180_x, g_ss180_y, g_ss180_z = make_oblique_gradients(g_ss180, ug_ss)

        # Slice-select direction spoilers
        g_ss_spoil = make_trapezoid(channel='z',system=system,area=g_ss.area*2,duration=3*pre_time)
        ##
        modify_gradient(g_ss_spoil,0)
        ##
        g_ss_spoil_x, g_ss_spoil_y, g_ss_spoil_z = make_oblique_gradients(g_ss_spoil, ug_ss)

    # Readout rewinder
    ro_pre_area = g_ro_neg.area / 2 if echo_type == 'gre' else g_ro_pos.area / 2
    g_ro_pre = make_trapezoid(channel='x',system=system, area=ro_pre_area, duration=pre_time)
    g_ro_pre_x, g_ro_pre_y, g_ro_pre_z = make_oblique_gradients(g_ro_pre,ug_fe)

    # Slice-selective rephasing
    g_ss_reph = make_trapezoid(channel='z',system=system,area=-g_ss.area/2,duration=pre_time)
    g_ss_reph_x, g_ss_reph_y, g_ss_reph_z = make_oblique_gradients(g_ss_reph, ug_ss)

    # Phase encode rewinder
    if echo_type == 'gre':
        pe_max_area = (Np/2)*delta_k_pe
    elif echo_type == 'se':
        pe_max_area = -(Np/2)*delta_k_pe

    g_pe_max = make_trapezoid(channel='y',system=system,area=pe_max_area,duration=pre_time)

    # Phase encoding blips
    dur = ceil(2 * sqrt(delta_k_pe/ system.max_slew) / 10e-6) * 10e-6
    g_blip = make_trapezoid(channel='y',system=system,area=delta_k_pe,duration=dur)

    # Delays
    duration_to_center = (Np/ 2 ) * calc_duration(g_ro_pos) + (Np-1) / 2 * calc_duration(g_blip) # why?

    if echo_type == 'se':
        delayTE1 = TE / 2 - calc_duration(g_ss) / 2 - pre_time - calc_duration(g_ss_spoil) - calc_duration(rf180) / 2
        delayTE2 = TE / 2 - calc_duration(rf180) / 2 - calc_duration(g_ss_spoil) - duration_to_center
        delay1 = make_delay(delayTE1)
        delay2 = make_delay(delayTE2)
    elif echo_type == 'gre':
        delayTE = TE - calc_duration(g_ss)/2 - pre_time - duration_to_center
        delay12 = make_delay(delayTE)

    delayTR = TR - TE - calc_duration(rf) / 2 - duration_to_center
    delay3 = make_delay(delayTR) # This might be different for each rep though. Fix later

#####################################################################################################
    # Multi-shot calculations
    ro_dirs = []
    ro_order = []

    # Find number of lines in each block

    if seg_type == 'blocked':

        # Number of lines in each full readout block
        nl = ceil(Np / n_shots)

        # Number of k-space lines per readout
        if Np%nl == 0:
            nlines_list = nl*np.ones(n_shots)
        else:
            nlines_list = nl*np.ones(n_shots-1)
            nlines_list = np.append(nlines_list,Np%nl)

        pe_scales = 2*np.append([0],np.cumsum(nlines_list)[:-1])/Np - 1
        g_blip_x, g_blip_y, g_blip_z = make_oblique_gradients(g_blip, ug_pe)
        for nlines in nlines_list:
            ro_dirs = np.append(ro_dirs, ((-1)**(np.arange(0,nlines)+1)+1)/2)


    elif seg_type == 'interleaved':
        # Minimum number of lines per readout
        nb = floor(Np / n_shots)

        # Number of k-space lines per readout
        nlines_list = np.ones(n_shots)*nb
        nlines_list[:Np%n_shots] += 1

        # Phase encoding scales (starts from -1; i.e. bottom left combined with pre-readout)
        pe_scales = 2*np.arange(0,(Np-n_shots)/Np,1/Np)[0:n_shots]-1
        print(pe_scales)
        # Larger blips
        modify_gradient(g_blip, scale=n_shots)
        g_blip_x, g_blip_y, g_blip_z = make_oblique_gradients(g_blip, ug_pe)

#        ro_order = np.reshape(np.reshape(np.arange(0,Np),(),order='F'),(0,Np))

        ro_order = np.zeros((nb+1,n_shots))
        ro_inds = np.arange(Np)
        # Readout order for recon
        for k in range(n_shots):
            cs = int(nlines_list[k])
            ro_order[:cs,k] = ro_inds[:cs]
            ro_inds = np.delete(ro_inds,range(cs))
        ro_order = ro_order.flatten()[:Np].astype(int)

        np.save('readout_order_for_interleaving.npy', ro_order)

        print(ro_order)

        # Readout directions in original (interleaved) order
        for nlines in nlines_list:
            ro_dirs = np.append(ro_dirs, ((-1)**(np.arange(0,nlines)+1)+1)/2)



#####################################################################################################

    # Add blocks

    for u in range(len(slice_locs)): # For each slice
        # Offset rf
        rf.freq_offset = g_ss.amplitude * slice_locs[u]
        for v in range(n_shots):
            # Find init. phase encode
            g_pe = copy.deepcopy(g_pe_max)
            modify_gradient(g_pe, pe_scales[v])
            g_pe_x, g_pe_y, g_pe_z = make_oblique_gradients(g_pe, ug_pe)
            # First RF
            seq.add_block(rf, g_ss_x, g_ss_y, g_ss_z)
            # Pre-winder gradients
            pre_grads_list = [g_ro_pre_x, g_ro_pre_y, g_ro_pre_z,
                              g_pe_x, g_pe_y, g_pe_z,
                              g_ss_reph_x, g_ss_reph_y, g_ss_reph_z]
            gtx, gty, gtz = combine_trap_grad_xyz(pre_grads_list, system, pre_time)
            seq.add_block(gtx, gty, gtz)


            # 180 deg pulse and spoilers, only for Spin Echo
            if echo_type == 'se':
                # First delay
                seq.add_block(delay1)
                # Second RF : 180 deg with spoilers on both sides
                seq.add_block(g_ss_spoil_x, g_ss_spoil_y, g_ss_spoil_z)#why?
                seq.add_block(rf180, g_ss180_x, g_ss180_y, g_ss180_z)
                seq.add_block(g_ss_spoil_x, g_ss_spoil_y, g_ss_spoil_z)
                # Delay between rf180 and beginning of readout
                seq.add_block(delay2)
            # For gradient echo it's just a delay
            elif echo_type == 'gre':
                seq.add_block(delay12)



            # EPI readout with blips
            for i in range(int(nlines_list[v])):
                if i%2 == 0:
                    seq.add_block(g_ro_pos_x, g_ro_pos_y, g_ro_pos_z, adc) # ro line in the positive direction
                else:
                    seq.add_block(g_ro_neg_x, g_ro_neg_y, g_ro_neg_z, adc) # ro line backwards
                seq.add_block(g_blip_x, g_blip_y, g_blip_z) # blip

            seq.add_block(delay3)

    # Display 1 TR
    #seq.plot(time_range=(0, TR))

    if write:
        seq.write("epi_{}_FOV{:.0f}mm_Nf{:d}_Np{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:d}deg_{:d}shots.seq"\
                  .format(echo_type, fov*1000, Nf, Np, TE * 1000, TR * 1000, fa, n_shots))


    print('EPI sequence (oblique) constructed')
    return seq, ro_dirs, ro_order




def parse_enc(enc):
    """Helper function for decoding enc parameter

    Parameters
    ----------
    enc : str or array_like
        Inputted encoding scheme to parse
    Returns
    -------
    ug_fe : numpy.ndarray
        Length-3 vector of readout direction
    ug_pe : numpy.ndarray
        Length-3 vector of phase encoding direction
    ug_ss : numpy.ndarray
        Length-3 vector of slice selecting direction

    """
    if isinstance(enc, str):
        xyz_dict = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
        ug_fe = xyz_dict[enc[0]]
        ug_pe = xyz_dict[enc[1]]
        ug_ss = xyz_dict[enc[2]]
    else:
        ug_fe = np.array(enc[0])
        ug_pe = np.array(enc[1])
        ug_ss = np.array(enc[2])

        ug_fe = ug_fe / np.linalg.norm(ug_fe)
        ug_pe = ug_pe / np.linalg.norm(ug_pe)
        ug_ss = ug_ss / np.linalg.norm(ug_ss)

    print('ug_fe: ', ug_fe)
    print('ug_pe: ', ug_pe)
    print('ug_ss: ', ug_ss)

    return ug_fe, ug_pe, ug_ss


def make_oblique_gradients(gradient,unit_grad):
    """Helper function to make oblique gradients

    (Gx, Gy, Gz) are generated from a single orthogonal gradient
    and a direction indicated by unit vector

    Parameters
    ----------
    gradient : Gradient
        Pulseq gradient object
    unit_grad: array_like
        Length-3 unit vector indicating direction of resulting oblique gradient

    Returns
    -------
    ngx, ngy, ngz : Gradient
        Oblique gradients in x, y, and z directions

    """
    ngx = copy.deepcopy(gradient)
    ngy = copy.deepcopy(gradient)
    ngz = copy.deepcopy(gradient)

    modify_gradient(ngx, unit_grad[0],'x')
    modify_gradient(ngy, unit_grad[1],'y')
    modify_gradient(ngz, unit_grad[2],'z')

    return ngx, ngy, ngz

def modify_gradient(gradient,scale,channel=None):
    """Helper function to modify the strength and channel of an existing gradient

    Parameters
    ----------
    gradient : Gradient
        Pulseq gradient object to be modified
    scale : float
        Scalar to multiply the gradient strength by
    channel : str, optional {None, 'x','y','z'}
        Channel to switch gradient into
        Default is None which keeps the original channel

    """
    gradient.amplitude *= scale
    gradient.area *= scale
    if gradient.type == 'trap':
        gradient.flat_area *= scale
    if channel != None:
        gradient.channel = channel


def combine_trap_grad_xyz(gradients,system,dur):
    """Helper function that merges multiple gradients

    A list of gradients are combined into one set of 3 oblique gradients (Gx, Gy, Gz) with equivalent areas
    Note that the waveforms are not preserved : the outputs will always be trapezoidal gradients

    Parameters
    ----------
    gradients : list
        List of gradients to be combined; there can be any number of x, y, or z gradients
    system : Opts
        Pulseq object that indicates system constraints for gradient parameters
    dur : float
        Duration of the output oblique gradients

    Returns
    -------
    gtx, gty, gtz : Gradient
        Oblique pulseq gradients with equivalent areas to all input gradients combined

    """
    gx_area, gy_area, gz_area = (0,0,0)
    for g in gradients:
        if g.channel == 'x':
            gx_area += g.area
        elif g.channel == 'y':
            gy_area += g.area
        elif g.channel == 'z':
            gz_area += g.area


    gtx = make_trapezoid(channel='x',system=system,area=gx_area,duration=dur)
    gty = make_trapezoid(channel='y',system=system,area=gy_area,duration=dur)
    gtz = make_trapezoid(channel='z',system=system,area=gz_area,duration=dur)

    return gtx, gty, gtz
