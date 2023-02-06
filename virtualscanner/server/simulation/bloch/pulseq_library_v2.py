# Add in TSE and IRSE sequences
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.opts import Opts
from scipy.io import loadmat, savemat
import math



def write_irse_interleaved_split_gradient(n=256, fov=250e-3, thk=5e-3, fa=90,
                                     te=12e-3, tr=2000e-3, ti=150e-3, slice_locations=[0], enc='xyz'):
    """
    2D IRSE sequence with overlapping gradient ramps and interleaved slices

    Inputs
    ------
    n : integer
        Matrix size (isotropic)
    fov : float
        Field-of-View in [meters]
    thk : float
        Slice thickness in [meters]
    fa : float
        Flip angle in [degrees]
    te : float
        Echo Time in [seconds]
    tr : float
        Repetition Time in [seconds]
    ti : float
        Inversion Time in [seconds]
    slice_locations : array_like
        Array of slice locations from isocenter in [meters]
    enc : str
        Spatial encoding directions; 1st - readout; 2nd - phase encoding; 3rd - slice select
        Use str with any permutation of x, y, and z to obtain orthogonal slices
        e.g. The default 'xyz' means axial(z) slice with readout in x and phase encoding in y


    Returns
    -------
    seq : pypulseq.Sequence.sequence Sequence object
        Output sequence object. Can be saved with seq.write('file_name.seq')
    sl_order : numpy.ndarray
        Randomly generated slice order. Useful for reconstruction.

    """
    # =========
    # SYSTEM LIMITS
    # =========
    # Set the hardware limits and initialize sequence object
    dG = 250e-6  # Fixed ramp time for all gradients
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=100e-6, rf_dead_time=100e-6,
                  adc_dead_time=10e-6)
    seq = Sequence(system)

    # =========
    # TIME CALCULATIONS
    # =========
    readout_time = 6.4e-3 + 2 * system.adc_dead_time
    t_ex = 2.5e-3
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_ref = 2e-3
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    t_sp = 0.5 * (te - readout_time - t_refwd)
    t_spex = 0.5 * (te - t_exwd - t_refwd)
    fsp_r = 1
    fsp_s = 0.5

    # =========
    # ENCODING DIRECTIONS
    # ==========
    ch_ro = enc[0]
    ch_pe = enc[1]
    ch_ss = enc[2]

    # =========
    # RF AND GRADIENT SHAPES - BASED ON RESOLUTION REQUIREMENTS : kmax and Npe
    # =========

    # RF Phases
    rf_ex_phase = np.pi / 2
    rf_ref_phase = 0

    # Excitation phase (90 deg)
    flip_ex = 90 * np.pi / 180
    rf_ex, gz, _ = make_sinc_pulse(flip_angle=flip_ex, system=system, duration=t_ex, slice_thickness=thk,
                                   apodization=0.5, time_bw_product=4, phase_offset=rf_ex_phase, return_gz=True)
    gs_ex = make_trapezoid(channel=ch_ss, system=system, amplitude=gz.amplitude, flat_time=t_exwd, rise_time=dG)

    # Refocusing (same gradient & RF is used for initial inversion)
    flip_ref = fa * np.pi / 180
    rf_ref, gz, _ = make_sinc_pulse(flip_angle=flip_ref, system=system, duration=t_ref, slice_thickness=thk,
                                    apodization=0.5, time_bw_product=4, phase_offset=rf_ref_phase, use='refocusing',
                                    return_gz = True)
    gs_ref = make_trapezoid(channel=ch_ss, system=system, amplitude=gs_ex.amplitude, flat_time=t_refwd, rise_time=dG)

    ags_ex = gs_ex.area / 2
    gs_spr = make_trapezoid(channel=ch_ss, system=system, area=ags_ex * (1 + fsp_s), duration=t_sp, rise_time=dG)
    gs_spex = make_trapezoid(channel=ch_ss, system=system, area=ags_ex * fsp_s, duration=t_spex, rise_time=dG)

    delta_k = 1 / fov
    k_width = n * delta_k

    gr_acq = make_trapezoid(channel=ch_ro, system=system, flat_area=k_width, flat_time=readout_time, rise_time=dG)
    adc = make_adc(num_samples=n, duration=gr_acq.flat_time - 40e-6, delay=20e-6)
    gr_spr = make_trapezoid(channel=ch_ro, system=system, area=gr_acq.area * fsp_r, duration=t_sp, rise_time=dG)
    gr_spex = make_trapezoid(channel=ch_ro, system=system, area=gr_acq.area * (1 + fsp_r), duration=t_spex,
                             rise_time=dG)

    agr_spr = gr_spr.area
    agr_preph = gr_acq.area / 2 + agr_spr
    gr_preph = make_trapezoid(channel=ch_ro, system=system, area=agr_preph, duration=t_spex, rise_time=dG)

    phase_areas = (np.arange(n) - n / 2) * delta_k

    # Split gradients and recombine into blocks
    gs1_times = [0, gs_ex.rise_time]
    gs1_amp = [0, gs_ex.amplitude]
    gs1 = make_extended_trapezoid(channel=ch_ss, times=gs1_times, amplitudes=gs1_amp)

    gs2_times = [0, gs_ex.flat_time]
    gs2_amp = [gs_ex.amplitude, gs_ex.amplitude]
    gs2 = make_extended_trapezoid(channel=ch_ss, times=gs2_times, amplitudes=gs2_amp)

    gs3_times = [0, gs_spex.rise_time, gs_spex.rise_time + gs_spex.flat_time,
                 gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time]
    gs3_amp = [gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude]
    gs3 = make_extended_trapezoid(channel=ch_ss, times=gs3_times, amplitudes=gs3_amp)

    gs4_times = [0, gs_ref.flat_time]
    gs4_amp = [gs_ref.amplitude, gs_ref.amplitude]
    gs4 = make_extended_trapezoid(channel=ch_ss, times=gs4_times, amplitudes=gs4_amp)

    gs5_times = [0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time,
                 gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]
    gs5_amp = [gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0]
    gs5 = make_extended_trapezoid(channel=ch_ss, times=gs5_times, amplitudes=gs5_amp)

    gs7_times = [0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time,
                 gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]
    gs7_amp = [0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude]
    gs7 = make_extended_trapezoid(channel=ch_ss, times=gs7_times, amplitudes=gs7_amp)

    gr3 = gr_preph

    gr5_times = [0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time,
                 gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]
    gr5_amp = [0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude]
    gr5 = make_extended_trapezoid(channel=ch_ro, times=gr5_times, amplitudes=gr5_amp)

    gr6_times = [0, readout_time]
    gr6_amp = [gr_acq.amplitude, gr_acq.amplitude]
    gr6 = make_extended_trapezoid(channel=ch_ro, times=gr6_times, amplitudes=gr6_amp)

    gr7_times = [0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time,
                 gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]
    gr7_amp = [gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0]
    gr7 = make_extended_trapezoid(channel=ch_ro, times=gr7_times, amplitudes=gr7_amp)

    t_ex = gs1.t[-1] + gs2.t[-1] + gs3.t[-1]
    t_ref = gs4.t[-1] + gs5.t[-1] + gs7.t[-1] + readout_time
    t_end = gs4.t[-1] + gs5.t[-1]


    # Calculate maximum number of slices that can fit in one TR
    # Without spoilers on each side
    TE_prime = 0.5 * calc_duration(gs_ref) + ti + te + 0.5 * readout_time + np.max(
        [calc_duration(gs7), calc_duration(gr7)]) + \
               calc_duration(gs4) + calc_duration(gs5)

    ns_per_TR = np.floor(tr / TE_prime)
    print('Number of slices that can be accommodated = ' + str(ns_per_TR))


    # Lengthen TR to accommodate slices if needed, and display message
    n_slices = len(slice_locations)
    if (ns_per_TR < n_slices):
        print(f'TR too short, adapted to include all slices to: {n_slices * TE_prime + 50e-6} s')
        TR = round(n_slices * TE_prime + 50e-6, ndigits=5)
        print('New TR = ' + str(TR))
        ns_per_TR = np.floor(TR / TE_prime)
    if (n_slices < ns_per_TR):
        ns_per_TR = n_slices
    # randperm so that adjacent slices do not get excited one after the other
    sl_order = np.random.permutation(n_slices)

    print('Number of slices acquired per TR = ' + str(ns_per_TR))

    # Delays
    TI_fill = ti - (0.5 * calc_duration(gs_ref) + calc_duration(gs1) + 0.5 * calc_duration(gs2))
    delay_TI = make_delay(TI_fill)
    TR_fill = tr - ns_per_TR * TE_prime
    delay_TR = make_delay(TR_fill)


    for k_ex in range(n):
        phase_area = phase_areas[k_ex]
        gp_pre = make_trapezoid(channel=ch_pe, system=system, area=phase_area, duration=t_sp, rise_time=dG)
        gp_rew = make_trapezoid(channel=ch_pe, system=system, area=-phase_area, duration=t_sp, rise_time=dG)
        s_in_TR = 0

        for s in range(len(sl_order)):

            # rf_ex.freq_offset = gs_ex.amplitude * slice_thickness * (sl_order[s] - (n_slices - 1) / 2)
            # rf_ref.freq_offset = gs_ref.amplitude * slice_thickness * (sl_order[s] - (n_slices - 1) / 2)

            rf_ex.freq_offset = gs_ex.amplitude * slice_locations[sl_order[s]]
            rf_ref.freq_offset = gs_ref.amplitude * slice_locations[sl_order[s]]

            rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
            rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]

            # Inversion using refocusing pulse
            seq.add_block(gs_ref, rf_ref)
            seq.add_block(delay_TI)

            # SE portion
            seq.add_block(gs1)
            seq.add_block(gs2, rf_ex)
            seq.add_block(gs3, gr3)

            seq.add_block(gs4, rf_ref)

            seq.add_block(gs5, gr5, gp_pre)
            seq.add_block(gr6, adc)

            seq.add_block(gs7, gr7, gp_rew)

            seq.add_block(gs4)
            seq.add_block(gs5)
            s_in_TR += 1
            if (s_in_TR == ns_per_TR):
                seq.add_block(delay_TR)
                s_in_TR = 0

    # Check timing to make sure sequence runs on scanner
    seq.check_timing()

    return seq, sl_order

def write_tse(n=256, fov=250e-3, thk=5e-3, fa_exc=90, fa_ref=180,
              te=50e-3, tr=2000e-3, slice_locations=[0], turbo_factor=4, enc='xyz'):
    """
    2D TSE sequence with interleaved slices and user-defined turbo factor

    Inputs
    ------
    n : integer
        Matrix size (isotropic)
    fov : float
        Field-of-View in [meters]
    thk : float
        Slice thickness in [meters]
    fa_exc : float
        Initial excitation flip angle in [degrees]
    fa_ref : float
        All following flip angles for spin echo refocusing in [degrees]
    te : float
        Echo Time in [seconds]
    tr : float
        Repetition Time in [seconds]
    slice_locations : array_like
        Array of slice locations from isocenter in [meters]
    turbo_factor : integer
        Number of echoes per TR
    enc : str
        Spatial encoding directions; 1st - readout; 2nd - phase encoding; 3rd - slice select
        Use str with any permutation of x, y, and z to obtain orthogonal slices
        e.g. The default 'xyz' means axial(z) slice with readout in x and phase encoding in y

    Returns
    -------
    seq : pypulseq.Sequence.sequence Sequence object
        Output sequence object. Can be saved with seq.write('file_name.seq')
    pe_order : numpy.ndarray
        (turbo_factor) x (number_of_excitations) matrix of phase encoding order
        This is required for phase sorting before ifft2 reconstruction.

    """
    # Set system limits
    ramp_time = 250e-6  # Ramp up/down time for all gradients where this is specified
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=100e-6,  # changed from 30e-6
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    # Initialize sequence
    seq = Sequence(system)

    # Spatial encoding directions
    ch_ro = enc[0]
    ch_pe = enc[1]
    ch_ss = enc[2]


    # Derived parameters
    n_slices = len(slice_locations)
    Nf, Np = (n,n)
    delta_k = 1 / fov
    k_width = Nf * delta_k

    # Number of echoes per excitation (i.e. turbo factor)
    n_echo = turbo_factor

    # Readout duration
    readout_time = 6.4e-3 + 2 * system.adc_dead_time

    # Excitation pulse duration
    t_ex = 2.5e-3
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time

    # Refocusing pulse duration
    t_ref = 2e-3
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time

    # Time gaps for spoilers
    t_sp = 0.5 * (te - readout_time - t_refwd)  # time gap between pi-pulse and readout
    t_spex = 0.5 * (te - t_exwd - t_refwd)  # time gap between pi/2-pulse and pi-pulse

    # Spoiling factors
    fsp_r = 1  # in readout direction per refocusing
    fsp_s = 0.5  # in slice direction per refocusing

    # 3. Calculate sequence components

    ## Slice-selective RF pulses & gradient
    #* RF pulses with zero frequency shift are created. The frequency is then changed before adding the pulses to sequence blocks for each slice.

    # 90 deg pulse (+y')
    rf_ex_phase = np.pi / 2
    flip_ex = fa_exc * np.pi / 180
    rf_ex, g_ss, _ = make_sinc_pulse(flip_angle=flip_ex, system=system, duration=t_ex, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=4, phase_offset=rf_ex_phase, return_gz = True)
    gs_ex = make_trapezoid(channel=ch_ss, system=system, amplitude=g_ss.amplitude, flat_time=t_exwd,
                           rise_time=ramp_time)

    # 180 deg pulse (+x')
    rf_ref_phase = 0
    flip_ref = fa_ref * np.pi / 180
    rf_ref, gz, _ = make_sinc_pulse(flip_angle=flip_ref, system=system, duration=t_ref, slice_thickness=thk,
                                    apodization=0.5, time_bw_product=4, phase_offset=rf_ref_phase,
                                    use='refocusing', return_gz = True)
    gs_ref = make_trapezoid(channel=ch_ss, system=system, amplitude=gs_ex.amplitude, flat_time=t_refwd,
                            rise_time=ramp_time)

    rf_ex, g_ss, _ = make_sinc_pulse(flip_angle=flip_ex, system=system, duration=t_ex, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=4, phase_offset=rf_ex_phase, return_gz=True)

    ## Make gradients and ADC
    # gs_spex : slice direction spoiler between initial excitation and 1st 180 pulse
    # gs_spr : slice direction spoiler between 180 pulses
    # gr_spr : readout direction spoiler; area is (fsp_r) x (full readout area)

    # SS spoiling
    ags_ex = gs_ex.area / 2
    gs_spr = make_trapezoid(channel=ch_ss, system=system, area=ags_ex * (1 + fsp_s), duration=t_sp, rise_time=ramp_time)
    gs_spex = make_trapezoid(channel=ch_ss, system=system, area=ags_ex * fsp_s, duration=t_spex, rise_time=ramp_time)

    # Readout gradient and ADC
    gr_acq = make_trapezoid(channel=ch_ro, system=system, flat_area=k_width, flat_time=readout_time,
                            rise_time=ramp_time)

    # No need for risetime delay since it is set at beginning of flattime; delay is ADC deadtime
    adc = make_adc(num_samples=Nf, duration=gr_acq.flat_time - 40e-6, delay=20e-6)

    # RO spoiling
    gr_spr = make_trapezoid(channel=ch_ro, system=system, area=gr_acq.area * fsp_r, duration=t_sp, rise_time=ramp_time)

    # Following is not used anywhere
    # gr_spex = make_trapezoid(channel=ch_ro, system=system, area=gr_acq.area * (1 + fsp_r), duration=t_spex, rise_time=ramp_time)

    # Prephasing gradient in RO direction
    agr_preph = gr_acq.area / 2 + gr_spr.area
    gr_preph = make_trapezoid(channel=ch_ro, system=system, area=agr_preph, duration=t_spex, rise_time=ramp_time)

    # Phase encoding areas
    # Need to export the pe_order for reconsturuction

    # Number of readouts/echoes to be produced per TR
    n_ex = math.floor(Np / n_echo)
    pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
    if divmod(n_echo, 2)[1] == 0:  # If there is an even number of echoes
        pe_steps = np.roll(pe_steps, -round(n_ex / 2))

    pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

    savemat('pe_info.mat', {'order': pe_order, 'dims': ['n_echo', 'n_ex']})
    phase_areas = pe_order * delta_k

    # Split gradients and recombine into blocks

    # gs1 : ramp up of gs_ex
    gs1_times = [0, gs_ex.rise_time]
    gs1_amp = [0, gs_ex.amplitude]
    gs1 = make_extended_trapezoid(channel=ch_ss, times=gs1_times, amplitudes=gs1_amp)

    # gs2 : flat part of gs_ex
    gs2_times = [0, gs_ex.flat_time]
    gs2_amp = [gs_ex.amplitude, gs_ex.amplitude]
    gs2 = make_extended_trapezoid(channel=ch_ss, times=gs2_times, amplitudes=gs2_amp)

    # gs3 : Bridged slice pre-spoiler
    gs3_times = [0, gs_spex.rise_time, gs_spex.rise_time + gs_spex.flat_time,
                 gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time]
    gs3_amp = [gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude]
    gs3 = make_extended_trapezoid(channel=ch_ss, times=gs3_times, amplitudes=gs3_amp)

    # gs4 : Flat slice selector for pi-pulse
    gs4_times = [0, gs_ref.flat_time]
    gs4_amp = [gs_ref.amplitude, gs_ref.amplitude]
    gs4 = make_extended_trapezoid(channel=ch_ss, times=gs4_times, amplitudes=gs4_amp)

    # gs5 : Bridged slice post-spoiler
    gs5_times = [0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time,
                 gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]
    gs5_amp = [gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0]
    gs5 = make_extended_trapezoid(channel=ch_ss, times=gs5_times, amplitudes=gs5_amp)

    # gs7 : The gs3 for next pi-pulse
    gs7_times = [0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time,
                 gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]
    gs7_amp = [0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude]
    gs7 = make_extended_trapezoid(channel=ch_ss, times=gs7_times, amplitudes=gs7_amp)

    # gr3 : pre-readout gradient
    gr3 = gr_preph

    # gr5 : Readout post-spoiler
    gr5_times = [0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time,
                 gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]
    gr5_amp = [0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude]
    gr5 = make_extended_trapezoid(channel=ch_ro, times=gr5_times, amplitudes=gr5_amp)

    # gr6 : Flat readout gradient
    gr6_times = [0, readout_time]
    gr6_amp = [gr_acq.amplitude, gr_acq.amplitude]
    gr6 = make_extended_trapezoid(channel=ch_ro, times=gr6_times, amplitudes=gr6_amp)

    # gr7 : the gr3 for next repeat
    gr7_times = [0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time,
                 gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]
    gr7_amp = [gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0]
    gr7 = make_extended_trapezoid(channel=ch_ro, times=gr7_times, amplitudes=gr7_amp)



    # Timing (delay) calculations

    # delay_TR : delay at the end of each TSE pulse train (i.e. each TR)
    t_ex = gs1.t[-1] + gs2.t[-1] + gs3.t[-1]
    t_ref = gs4.t[-1] + gs5.t[-1] + gs7.t[-1] + readout_time
    t_end = gs4.t[-1] + gs5.t[-1]
    TE_train = t_ex + n_echo * t_ref + t_end
    TR_fill = (tr - n_slices * TE_train) / n_slices
    TR_fill = system.grad_raster_time * round(TR_fill / system.grad_raster_time)
    if TR_fill < 0:
        TR_fill = 1e-3
        print(f'TR too short, adapted to include all slices to: {1000 * n_slices * (TE_train + TR_fill)} ms')
    else:
        print(f'TR fill: {1000 * TR_fill} ms')
    delay_TR = make_delay(TR_fill)


    # Add building blocks to sequence

    for k_ex in range(n_ex + 1):  # For each TR
        for s in range(n_slices):  # For each slice (multislice)

            if slice_locations is None:
                rf_ex.freq_offset = gs_ex.amplitude * thk * (s - (n_slices - 1) / 2)
                rf_ref.freq_offset = gs_ref.amplitude * thk * (s - (n_slices - 1) / 2)
                rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
                rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]
            else:
                rf_ex.freq_offset = gs_ex.amplitude * slice_locations[s]
                rf_ref.freq_offset = gs_ref.amplitude * slice_locations[s]
                rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
                rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]

            seq.add_block(gs1)
            seq.add_block(gs2, rf_ex)  # make sure gs2 has channel ch_ss
            seq.add_block(gs3, gr3)

            for k_echo in range(n_echo):  # For each echo
                if k_ex > 0:
                    phase_area = phase_areas[k_echo, k_ex - 1]
                else:
                    # First TR is skipped so zero phase encoding is needed
                    phase_area = 0.0  # 0.0 and not 0 because -phase_area should successfully result in negative zero

                gp_pre = make_trapezoid(channel=ch_pe, system=system, area=phase_area, duration=t_sp,
                                        rise_time=ramp_time)
                # print('gp_pre info: ', gp_pre)

                gp_rew = make_trapezoid(channel=ch_pe, system=system, area=-phase_area, duration=t_sp,
                                        rise_time=ramp_time)

                seq.add_block(gs4, rf_ref)
                seq.add_block(gs5, gr5, gp_pre)

                # Skipping first TR
                if k_ex > 0:
                    seq.add_block(gr6, adc)
                else:
                    seq.add_block(gr6)

                seq.add_block(gs7, gr7, gp_rew)

            seq.add_block(gs4)
            seq.add_block(gs5)
            seq.add_block(delay_TR)

    # Check timing to make sure sequence runs on scanner
    seq.check_timing()

    return seq, pe_order




if __name__ == '__main__':
    # Make sequences with the seq validation default settings, check timing, visualize, and export

    # Define ACR slice locations
    # n_slices = 11
    # thk = 5e-3
    # gap = 5e-3
    # L = (n_slices - 1) * (thk + gap)
    # # displacement = -4.4e-3
    # displacement = -4e-3
    # acr_sl_locs = displacement + np.arange(-L / 2, L / 2 + thk + gap, thk + gap)
    # print('Slice locations in meters: ', acr_sl_locs)
    #
    #
    # # Make, check, visualize, and save an IRSE sequence
    # # IRSE
    # # seq_irse, sl_order = write_irse_interleaved_split_gradient(slice_locations=acr_sl_locs)
    # # print(seq_irse.test_report())
    # # seq_irse.plot(time_range=[0,2])
    # #
    # # seq_irse.write('irse.seq')
    # # savemat('irse_info.mat',{'sl_order':sl_order})
    #
    # # TSE
    # # Make, check, visualize, and save a TSE sequence
    # seq_tse, pe_order = write_tse(slice_locations=acr_sl_locs)
    # print(seq_tse.test_report())
    # seq_tse.plot(time_range=[0,2])
    #
    # seq_tse.write('tse.seq')
    # savemat('tse_info.mat',{'pe_order':pe_order})

    seq, pe_order = write_tse(n=64, fov=10e-3, thk=5e-3, fa_exc=90, fa_ref=180,
              te=50e-3, tr=2000e-3, slice_locations=[0], turbo_factor=4, enc='xyz')
    seq.write('tse64.seq')
