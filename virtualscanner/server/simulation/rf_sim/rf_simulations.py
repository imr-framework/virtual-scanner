# What functionalities should rf_simulations.py have?
# - simulate sinc, block, and gaussian pulses with any flip angle/bandwidth
# - simulate SLR pulses and pulses from the pulseq rf constructor
# - simulate adiabatic pulses: half passage, full passage, BIR-1, BIR-4, hyperbolic secant
# - connect to animate_spins.py for visualization (can integrated into virtual scanner later)
################################################################################
# Simulates effects of some RF pulses & plots their instantaneous & average power
import virtualscanner.server.simulation.bloch.spingroup_ps as sg
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from virtualscanner.server.simulation.rf_sim.animate_spins import animate_spins
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.opts import Opts
from virtualscanner.server.simulation.rf_sim.rf_helpers import *
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from scipy.io import savemat



# Constants
GAMMA = 42.58e6 * 2 * pi

# Caller function
def simulate_rf(bw_spins, n_spins, pdt1t2, flip_angle, dt, pulse_type, dur = 0, solver="Euler", display=True, **kwargs):
    '''Simulates and plots absolute signal (|Mxy|) of isochromats over time

    Parameters
    ----------
    bw_spins : float
        Bandwidth of simulated spins in Hz
    n_spins : int
        Number of spins simulated, with a linear spread of frequencies
    pdt1t2 : array_like
        Length-3 list of [PD, T1, T2] values. T1 and T2 should be in seconds
    flip_angle : float
        Flip angle in degrees
    dt : float
        Time discretization in seconds
    pulse_type : string {'block','sinc','gaussian','custom'}
        Type of RF pulse
    dur : float, optional
        Duration of RF pulse; applies to block, sinc, and gaussian
    solver : str, optional
        Type of numerical solver to use;
        "Euler" - Simple, uniform time stepping (default)
        "RK45" - Explicit Runge-Kutta method of order 5(4) (see scipy.integrate.solve_ivp)
    **kwargs
        Optional named parameters passed into make_rf_shapes

    '''


    # Create list of spins
    df_array = np.linspace(-bw_spins/2, bw_spins/2, n_spins)

    if solver=="Euler":
        spins = [sg.SpinGroup(loc=(0,0,0), pdt1t2=pdt1t2, df=this_df) for this_df in df_array]
    else:
        spins = [sg.NumSolverSpinGroup(loc=(0,0,0),pdt1t2=pdt1t2, df=this_df) for this_df in df_array]

    tmodel, pulse_shape = make_rf_shapes(pulse_type=pulse_type,flip_angle=flip_angle, dt=dt, dur=dur, **kwargs)

    # No gradients - we are using the off-resonance parameter for making off-center spins
    grads_shape = np.zeros((3, len(pulse_shape)))

    # Simulate
    all_results = [spin.apply_rf_store(pulse_shape, grads_shape, dt) for spin in spins]
    #all_ms = [spin.apply_rf_solveivp_store(pulse_shape, grads_shape, dt)[0] for spin in spins]
    all_signals = np.array([result[0] for result in all_results])
    all_magnetizations = np.array([result[1] for result in all_results])

    fig = plt.figure(1)
    ax = fig.add_axes(xlim=(0, 4), ylim=(-2, 2))
    for a in range(len(spins)):
        pp = plt.plot(tmodel, np.absolute(all_signals[a]))


    # Bandwidth info to print out
    if 'bw_rf' in kwargs:
        bw_info = '(BW='+ str(kwargs['bw_rf']) + ' Hz)'
    else:
        bw_info=''

    if display:
        plt.title(pulse_type + ' RF pulse ' + bw_info + ' applied to isochromats')
        plt.legend(['df = '+ str(df) + ' Hz' for df in df_array])

        plt.xlabel('Time (s)')
        plt.ylabel('Mxy (a.u.)')
        plt.show()


        mzs = np.squeeze(all_magnetizations[:,2,:])
        fig2 = plt.figure(2)

        for b in range(len(spins)):
            plt.plot(tmodel, mzs[b,:])

        plt.title(pulse_type + ' RF pulse ' + bw_info + ' applied to isochromats')
        plt.legend(['df = '+ str(df) + ' Hz' for df in df_array])

        plt.xlabel('Time (s)')
        plt.ylabel('Mz (a.u.)')
        plt.show()


    return all_signals, all_magnetizations


def animate_rf_action(bw_spins, n_spins, pdt1t2, pulse_type, flip_angle, dt=1e-6, dur = 0, solver="Euler",
                      save_fig=False, acc_factor=1, **kwargs):
    """Generate 3D animation of spins under the influence of RF pulse and T1, T2 relaxation

    Parameters
    ----------
    bw_spins : float
        Bandwidth of isochromats simulated, in Hz
    n_spins : int
        Number of linearly spaced isochromats
    pdt1t2 : array_like
        Length-3 list of [PD, T1, T2] values. T1 and T2 should be in seconds
    pulse_type : string {'block','sinc','gaussian','custom'}
        Type of RF pulse
    flip_angle : float
        Flip angle in degrees
    dt : float
        Time discretization in seconds
    dur : float, optional
        Duration of RF pulse; applies to block, sinc, and gaussian
    solver : str, optional
        "Euler" : default time stepping
        "RK45" : Explicit Runge-Kutta Method of Order 4(5)
    save_fig : bool, optional
        Whether or not to save the animation as a gif file. Default is False.
    acc_factor : int, optional
        Acceleration factor for displaying the animation. Default is 1 (no acceleration).
        Specifically, (acc_factor - 1) frames are skipped in every (acc_factor frames).
    **kwargs
        Optional named parameters passed into make_rf_shapes
    """

    all_ms = []
    df_array = np.linspace(-bw_spins/2, bw_spins/2, n_spins)

    spins = []
    if solver == "Euler":
        spins = [sg.SpinGroup(loc=(0,0,0), pdt1t2=pdt1t2, df=df_array[k]) for k in range(n_spins)]
    elif solver == "RK45":
        spins = [sg.NumSolverSpinGroup(loc=(0,0,0),pdt1t2=pdt1t2,df=df_array[k])for k in range(n_spins)]

    tmodel, pulse_shape = make_rf_shapes(pulse_type, flip_angle, dt, dur, **kwargs)
    grads_shape = np.zeros((3, len(pulse_shape)))

    title = pulse_type + ' RF pulse with FA = ' + str(flip_angle)

    all_ms = np.array([np.transpose(spin.apply_rf_store(pulse_shape,grads_shape,dt)[1]) for spin in spins])
    animate_spins(all_ms, acc_factor=acc_factor, save_fig=save_fig, title=title)


def make_rf_shapes(pulse_type, flip_angle, dt, dur=0, tmodel=None, **kwargs):
    """
    Parameters
    ----------
    pulse_type : string {'block','sinc','gaussian','custom'}
        Type of RF pulse
    flip_angle : float
        Flip angle in degrees
    dt : float
        Time discretization in seconds
    dur : float, optional
        Duration of RF pulse

    Below are **kwargs options:
    nzc : int, optional
        Number of zero crossings; applies only to sinc and overrides dur
    bw_rf : float, optional
        RF bandwidth. Must be provided for gaussian and sinc pulses.
    bw_factor : float
        Used only for Gaussian pulse. Percentage of maximum used in
        defining bandwidth in frequency domain. Defaults to 0.5 (FWHM).
    pulse_shape : np.ndarray
        Used only for custom pulse - samples of B1+(t) in time.
        Specify only this and dt to have uniform time points generated.
    time_points : np.ndarray
        Used only for custom pulse - where B1+(t) is sampled in time.

    Returns
    -------
    tmodel : np.ndarray
        Time points of discretized RF pulse
    pulse_shape : np.ndarray
        Complex representation of RF waveform (B1+(t))
    """
    # Select pulse
    if pulse_type == 'block':
        # Construct with specified duration and flip angle
        tmodel = np.linspace(-dur / 2, dur / 2, dur / dt)
        b1 = (flip_angle * pi / 180) / (dur * GAMMA)
        pulse_shape = b1 * np.ones(len(tmodel))

    elif pulse_type == 'sinc':
        # Construct with specified bandwidth, duration, and flip angle
        if 'bw_rf' not in kwargs:
            raise ValueError('Bandwidth is not provided!')
        bw_rf = kwargs['bw_rf']

        if 'nzc' not in kwargs:
            print("#zero crossings not given - using the specified duration instead.")
            tmodel = np.linspace(-dur / 2, dur / 2, int(dur / dt))
        else:
            nzc = kwargs['nzc']
            tzc = 1 / bw_rf
            tmodel = np.linspace(-nzc * tzc / 2, nzc * tzc / 2, int(nzc * tzc / dt))

        pulse_shape = np.sin(pi * bw_rf * tmodel) / (pi * bw_rf * tmodel)
        b1 = (flip_angle * pi / 180) / (GAMMA * np.trapz(pulse_shape, tmodel))
        pulse_shape = b1 * pulse_shape

    elif pulse_type == 'gaussian':
        if 'bw_rf' not in kwargs:
            raise ValueError('Bandwidth is not provided!')

        bw_rf = kwargs['bw_rf']

        if 'bw_factor' not in kwargs:
            print("Bandwidth factor not given - using FWHM")
            bw_factor = 0.5 # FWHM
        else:
            bw_factor = kwargs['bw_factor']
            if bw_factor <= 0 or bw_factor >= 1:
                raise ValueError("Bandwidth factor should be positive and less than 1.")

        tmodel = np.linspace(-dur/2, dur/2, dur/dt)
        tau = np.sqrt(2*np.log(1/bw_factor)/(bw_rf*pi))
        gauss_shape = np.exp(-tmodel*tmodel/(2*(tau**2)))
        B1 = (flip_angle * pi / 180) / (GAMMA*np.trapz(gauss_shape, tmodel))
        pulse_shape = B1*gauss_shape

    elif pulse_type == 'custom':
        if 'pulse_shape' not in kwargs:
            raise ValueError("Pulse shape is not specified :(")

        pulse_shape = kwargs['pulse_shape']

        if 'time_points' not in kwargs:
            print("Time points not specified; using dt_rf instead")
            tmodel = dt*np.arange(0, len(pulse_shape))
        else:
            tmodel = kwargs['time_points']

    return tmodel, pulse_shape



if __name__ == '__main__':
    # Examples of simulation!
    pd = 1
    t1 = 0
    t2 = 0
    #t1 = 0.5
    #t2 = 0.05
    fa = 90
    dt_rf = 1e-6
    #simulate_rf(flip_angle=90, dt=1e-6, bw_spins=1e2, n_spins=7, pulse_type='block', dur = 10e-3)
  #  simulate_rf( bw_spins=5e3, n_spins=5, pdt1t2=(pd,t1,t2), flip_angle=90, dt=dt_rf, pulse_type='sinc', nzc=8, bw_rf=5e3)


    N = 100
 #   simulate_rf(bw_spins=200, n_spins=5, pdt1t2=(pd,t1,t2),flip_angle=N*90,dt=dt_rf,
 #               pulse_type='block', dur=0.5)

    #animate_rf_action(bw_spins=200,n_spins=5, pdt1t2=(pd,t1,t2),flip_angle=N*90,dt=dt_rf,dur=0.5,
   #                   pulse_type='block',save_fig=True,acc_factor=40)


    # Long block pulse (i.e. constant STAR)
    #animate_rf_action(bw_spins=1e3, n_spins=5, pdt1t2=(pd,t1,t2), flip_angle=fa, dt=dt_rf, dur=0.003,
     #                 pulse_type = 'block', save_fig=True, acc_factor=2)

    # Gaussian pulse
    #animate_rf_action(bw_spins=2e2, n_spins=5, pdt1t2=(pd,t1,t2), flip_angle=90, dt=dt_rf, dur=0.005,
     #                pulse_type='gaussian', bw_rf=5e3, save_fig=True, acc_factor=3)




    # Sinc pulse (no apodization)
 #   animate_rf_action(bw_spins=5e3, n_spins=5, pdt1t2=(pd,t1,t2), flip_angle=fa, dt=dt_rf, pulse_type='sinc', nzc=6, bw_rf=5e3,
  #                    save_fig=False, acc_factor=5)


    # This creates a pulseq sinc pulse (it has apodization)
    # system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
    #               slew_unit='T/m/s', rf_ringdown_time=30e-6,
    #               rf_dead_time=100e-6, adc_dead_time=20e-6)
    # flip = pi/2
    # thk = 5e-3
    # rf, g_ss, __ = make_sinc_pulse(flip_angle=flip, system=system, duration=4e-3, slice_thickness=thk,
    #                           apodization=0.5, time_bw_product=4, return_gz=True)


    # Simulate a basic sinc pulse (no apodization )
  #  a = simulate_rf(bw_spins=5e3, n_spins=5, pdt1t2=(pd, t1, t2), flip_angle=90, dt=dt_rf,
   #                     pulse_type='sinc', nzc=8, bw_rf=5e3)
   #
    #a = simulate_rf(bw_spins=25e3, n_spins=5, pdt1t2=(pd, t1, t2), flip_angle=90, dt=dt_rf,
     #                solver="RK45",
      #               pulse_type='sinc', nzc=8, bw_rf=5e3)

    # 073021 : Simulate the sinc pulse used in 3D UTE
    # system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
    #               rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    # slab_thk = 200e-3
    # ro_asymmetry = 0
    # rf_center = 0.5
    # FA = 5 # degrees
    # rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA * np.pi / 180, duration=1e-3, slice_thickness=slab_thk,
    #                                   apodization=0.5,
    #                                   time_bw_product=2, center_pos=rf_center, system=system, return_gz=True)
    # GAMMA_BAR = GAMMA/(2*np.pi)
    # rf_dt = rf.t[1] - rf.t[0]
    # print(f'Slice bw : {slab_thk*gz.amplitude} Hz')
    # signals, magnetizations = simulate_rf(bw_spins=slab_thk*gz.amplitude, n_spins=100, pdt1t2=(1,0,0), flip_angle=90, dt=rf_dt,
    #                   solver="RK45",
    #                   pulse_type='custom', pulse_shape=rf.signal/GAMMA_BAR, display=False)
    # print(magnetizations.shape)
    # savemat('3dUTE_pulse_final_magnetizations.mat',{'finals_magnetizations': magnetizations[:,:,-1]})


    # 080221 : Simulate the half-pulses used in 2D rw UTE  (pypulseq demo)
    # Without slice refocusing!!!!!!

    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

    # Sequence components
    FA = 10 # deg
    thk = 5e-3

    #rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA*np.pi/180, duration=1e-3, slice_thickness=thk, apodization=0.5,
    #                                  time_bw_product=2, center_pos=1, system=system, return_gz=True)

    rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA*np.pi/180, duration=2e-3, slice_thickness=thk, apodization=0.5,
                                      time_bw_product=2, center_pos=0.5, system=system, return_gz=True)
    GAMMA_BAR = GAMMA/(2*np.pi)
    rf_dt = rf.t[1] - rf.t[0]
    print(f'Slice bw : {thk*gz.amplitude} Hz')
    bwbw = 2*thk*gz.amplitude
    signals, m = simulate_rf(bw_spins=bwbw, n_spins=200, pdt1t2=(1,0,0), flip_angle=90, dt=rf_dt,
                       solver="RK45",
                       pulse_type='custom', pulse_shape=rf.signal/GAMMA_BAR, display=False)
    print(m.shape)


    savemat('sim_results/2d_rw_UTE_fullpulse_matched_results.mat', {'m': m[:, :, -1], 'thk_sim': 2 * thk})