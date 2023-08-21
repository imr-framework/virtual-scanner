from virtualscanner.server.simulation.bloch.spingroup_ps_t2star import *
import matplotlib.pyplot as plt
# Test 1: different number of spins
# Generate a spin echo. Compare spins with / without T2star modeling


#REF: applying RF

#isc.apply_rf(pulse_shape=cpars[0], grads_shape=cpars[1], dt=cpars[2])\

###

def run_simulation_time_resolved(t2=0.2,t2star=0.01,ideal_RF=False, n=100, dt=1e-4, display=True):
    #print(f'Using {n} spins for T2 star model')
    TE = 200e-3 # 200
    system = Opts()
    rf90 = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=2e-3,
                           slice_thickness=5e-3, apodization=0.5, time_bw_product=4)
    rf180 = make_sinc_pulse(flip_angle=np.pi, system=system, duration=2e-3,
                           slice_thickness=5e-3, apodization=0.5, time_bw_product=4)
    seq = Sequence()
    seq.add_block(rf90)
    seq.add_block(rf180)

    rf_info = blcsim.store_pulseq_commands(seq)
    pars90 = rf_info['params'][0]
    pars180= rf_info['params'][1]

    delay1 = make_delay((TE/2) - 0.5*calc_duration(rf90) - 0.5*calc_duration(rf180))
    delay2 = make_delay((TE/2) - 0.5*calc_duration(rf180))
    delay3 = make_delay(TE - 0.5*calc_duration(rf90))

    sg0 = SpinGroupT2star(loc=(0, 0, 0), pdt1t2=(1, 2, t2), t2star=0, num_spins=n)
    sg1 = SpinGroupT2star(loc=(0, 0, 0), pdt1t2=(1, 2, t2), t2star=t2star, num_spins=n)

    # blcsim.apply_pulseq_commands(sg0, seq_info_SE, store_m=False)

    if not ideal_RF:
        #print("Using PyPulseq sinc pulses")
        tmodel_SE = pars90[2] * np.arange(len(pars90[0]))
        # SE
        s0_SE, __ = sg0.apply_rf_store(pulse_shape=pars90[0], grads_shape=pars90[1], dt=pars90[2])
        s1_SE, __ = sg1.apply_rf_store(pulse_shape=pars90[0], grads_shape=pars90[1], dt=pars90[2])

        # Simulate in time!
        ct = 0
        while ct <= delay1.delay:
            sg0.delay(dt)
            sg1.delay(dt)
            s0_SE = np.append(s0_SE,sg0.get_m_signal())
            s1_SE = np.append(s1_SE,sg1.get_m_signal())
            tmodel_SE = np.append(tmodel_SE, tmodel_SE[-1]+dt)
            ct += dt

        sg0_SE_180, __ = sg0.apply_rf_store(pulse_shape=pars180[0], grads_shape=pars180[1], dt=pars180[2])
        sg1_SE_180, __ = sg1.apply_rf_store(pulse_shape=pars180[0], grads_shape=pars180[1], dt=pars180[2])
        s0_SE = np.append(s0_SE, sg0_SE_180)
        s1_SE = np.append(s1_SE, sg1_SE_180)
        tmodel_SE = np.append(tmodel_SE, tmodel_SE[-1]+pars180[2] * np.arange(len(pars180[0])))

        ct = 0
        while ct <= delay2.delay:
            sg0.delay(dt)
            sg1.delay(dt)
            s0_SE = np.append(s0_SE,sg0.get_m_signal())
            s1_SE = np.append(s1_SE,sg1.get_m_signal())
            tmodel_SE = np.append(tmodel_SE, tmodel_SE[-1]+dt)
            ct += dt

        ###############################################

        # Second, GRE
        sg0.reset()
        sg1.reset()

        tmodel_GRE = pars90[2] * np.arange(len(pars90[0]))
        # SE
        s0_GRE, __ = sg0.apply_rf_store(pulse_shape=pars90[0], grads_shape=pars90[1], dt=pars90[2])
        s1_GRE, __ = sg1.apply_rf_store(pulse_shape=pars90[0], grads_shape=pars90[1], dt=pars90[2])

        ct = 0
        while ct <= delay3.delay:
            sg0.delay(dt)
            sg1.delay(dt)
            s0_GRE = np.append(s0_GRE, sg0.get_m_signal())
            s1_GRE = np.append(s1_GRE, sg1.get_m_signal())
            tmodel_GRE = np.append(tmodel_GRE, tmodel_GRE[-1] + dt)
            ct += dt


    else:
        #print("Using ideal RF rotations")
        tmodel_SE = np.array([0])
        # 90
        sg0.set_m(np.array([[1], [0], [0]]))
        sg1.set_m(np.array([[1], [0], [0]]))

        #
        s0_SE = np.array(sg0.get_m_signal())
        s1_SE = np.array(sg1.get_m_signal())

        # Delay 1
        ct = 0
        while ct <= TE/2:
            sg0.delay(dt)
            sg1.delay(dt)
            s0_SE = np.append(s0_SE, sg0.get_m_signal())
            s1_SE = np.append(s1_SE, sg1.get_m_signal())
            tmodel_SE = np.append(tmodel_SE, tmodel_SE[-1]+dt)
            ct += dt

        # 180
        for spin in sg0.spin_list:
            spin.m[1] = -spin.m[1]
        for spin in sg1.spin_list:
            spin.m[1] = -spin.m[1]

        s0_SE = np.append(s0_SE, np.array(sg0.get_m_signal()))
        s1_SE = np.append(s1_SE, np.array(sg1.get_m_signal()))
        tmodel_SE = np.append(tmodel_SE, tmodel_SE[-1])

        # Delay 2
        ct = 0
        while ct <= TE/2:
            sg0.delay(dt)
            sg1.delay(dt)
            s0_SE = np.append(s0_SE, sg0.get_m_signal())
            s1_SE = np.append(s1_SE, sg1.get_m_signal())
            tmodel_SE = np.append(tmodel_SE, tmodel_SE[-1]+dt)
            ct += dt

        ################################
        # GRE
        tmodel_GRE = np.array([0])
        # 90
        sg0.reset()
        sg1.reset()
        sg0.set_m(np.array([[1], [0], [0]]))
        sg1.set_m(np.array([[1], [0], [0]]))

        #
        s0_GRE = np.array(sg0.get_m_signal())
        s1_GRE = np.array(sg1.get_m_signal())

        # Delay 1
        ct = 0
        while ct <= TE:
            sg0.delay(dt)
            sg1.delay(dt)
            s0_GRE = np.append(s0_GRE, sg0.get_m_signal())
            s1_GRE = np.append(s1_GRE, sg1.get_m_signal())
            tmodel_GRE = np.append(tmodel_GRE, tmodel_GRE[-1]+dt)
            ct += dt

    results = {'s0_SE': s0_SE, 's1_SE':s1_SE, 's0_GRE':s0_GRE, 's1_GRE':s1_GRE,
               'tmodel_SE':tmodel_SE, 'tmodel_GRE':tmodel_GRE}
    if display:
        se_rf90_time = 0.5*calc_duration(rf90)
        se_rf180_time = calc_duration(rf90) + delay1.delay + 0.5*calc_duration(rf180)
        gre_rf90_time = 0.5*calc_duration(rf90)

        plt.figure(1)
        plt.subplot(211)
        plt.title('Spin Echo (magnitude)')
        plt.plot(1e3*tmodel_SE,np.absolute(s0_SE),'-k',label="No T2 star")
        plt.plot(1e3*tmodel_SE,np.absolute(s1_SE),'-r',label="With T2 star")
        plt.plot(1e3*se_rf90_time*np.array([1,1]),[-0.5,1.5],'-c',label="RF pulse (90 deg)")
        plt.plot(1e3*se_rf180_time*np.array([1,1]),[-0.5,1.5],'-b',label="RF pulse (180 deg)")
        plt.xlim([-5, 1e3*TE])
        plt.ylim([0,1])

        plt.xlabel("Time (ms)")
        plt.ylabel("Signal (a.u.) (max = 1.0)")
        plt.legend()
        plt.subplot(212)
        plt.title('Spin Echo (phase)')
        plt.plot(1e3*tmodel_SE,np.angle(s0_SE),'--k',label="No T2 star")
        plt.plot(1e3*tmodel_SE,np.angle(s1_SE),'--r',label="With T2 star")
        plt.plot(1e3*se_rf90_time * np.array([1, 1]), [-4,4], '-c', label="RF pulse (90 deg)")
        plt.plot(1e3*se_rf180_time * np.array([1, 1]),[-4,4], '-b', label="RF pulse (180 deg)")
        plt.xlim([-5, 1e3*TE])
        plt.ylim([-np.pi, np.pi])
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (radians)")
        plt.legend()


        plt.figure(2)
        plt.subplot(211)
        plt.title('Gradient Recalled Echo (magnitude)')
        plt.plot(1e3*tmodel_GRE,np.absolute(s0_GRE),'-k',label="No T2 star")
        plt.plot(1e3*tmodel_GRE,np.absolute(s1_GRE),'-g',label="With T2 star")
        plt.plot(1e3*gre_rf90_time*np.array([1,1]),[-0.5,1.5],'-c',label="RF pulse (90 deg)")
        plt.xlim([-5, 1e3*TE])
        plt.ylim([0, 1])
        plt.xlabel("Time (ms)")
        plt.ylabel("Signal (a.u.) (max = 1.0)")
        plt.legend()
        plt.subplot(212)
        plt.title('Gradient Recalled Echo (phase)')
        plt.plot(1e3*tmodel_GRE,np.angle(s0_GRE),'--k',label="No T2 star")
        plt.plot(1e3*tmodel_GRE,np.angle(s1_GRE),'--g',label="With T2 star")
        plt.plot(1e3*gre_rf90_time*np.array([1,1]),[-4,4],'-c',label="RF pulse (90 deg)")
        plt.xlim([-5,1e3*TE])
        plt.ylim([-np.pi,np.pi])
        plt.xlabel("Time (ms)")
        plt.ylabel("Phase (radians)")
        plt.legend()
        plt.show()

    final_signals = np.absolute(np.array([s0_SE[-1],s1_SE[-1],s0_GRE[-1],s1_GRE[-1]]))

    # Print final values
    #print(f'T2 = {t2*1e3} ms; T2 star = {t2star*1e3} ms')
    #print(f'SE: {np.absolute(s0_SE[-1])} for no T2 star, {np.absolute(s1_SE[-1])} for T2 star; \
       #      GRE: {np.absolute(s0_GRE[-1])} for no T2 star, {np.absolute(s1_GRE[-1])} for T2 star.')

    return results, final_signals

def run_t2star_sim_histograms(num_spins=[4,20,100],repeats=50,nbins=5,use_ideal=True):
    t2 = 0.2
    t2star = 0.1
    names = ['SE_s0','SE_s1','GRE_s0','GRE_s1']

    results_for_hist = np.zeros((len(num_spins),repeats,4))

    TE = 200e-3

    for u in range(len(num_spins)):
        for i in range(repeats):
            __, fin_sig = run_simulation_time_resolved(t2=t2,t2star=t2star,ideal_RF=use_ideal,
                                                       n=num_spins[u], dt=1e-4, display=False)
            results_for_hist[u,i,:] = fin_sig

    # Plot true value vs. histogram
    plt.figure()
    q = 0

    predicted_values = np.array([np.exp(-TE/t2), np.exp(-TE/t2), np.exp(-TE/t2), np.exp(-TE/t2star)])

    for u in range(len(num_spins)):
        for j in range(4):
            plt.subplot(len(num_spins),4,q+1)
            plt.hist(np.squeeze(results_for_hist[u,:,j]),bins=nbins)
            plt.vlines(x=predicted_values[j],ymin=0,ymax=repeats,colors='red')
            plt.title(f"{names[j]} (n={num_spins[u]})")
            q += 1

    plt.show()

    return results_for_hist



if __name__ == '__main__':
    #run_simulation_time_resolved(t2=0.2,t2star=0.1,ideal_RF=True, n=20, dt=1e-4, display=True)
    results = run_t2star_sim_histograms(use_ideal=False) # using Ideal RF
    savemat('../rf_sim/msi/t2star_hist_data_pypulseq.mat', {'results': results})
