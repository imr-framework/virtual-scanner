import bloch as blc
import numpy as np
import copy

class BlochSequence(): # this is blc Sequence, not pulseq Sequence
    """
    Generic class for MR pulse sequences
    Different from the Sequence class in pulseq!!!!!!
    """
    GAMMA_BAR = 42.58e6
    GAMMA = 42.58e6 * 2 * np.pi

    def __init__(self, name="Pulse Sequence"):
        self._name = name
        self._events = []

    def get_events(self):
        return self._events

    def plot_psd(self):
        print(self._events)


class GRESequence(BlochSequence):
    """
    Basic gradient-echo sequence with 2D encoding (phase & frequency)
        for testing out bloch simulation
    inputs: tr (s), te (s), flip_angle (rad), fov = (fov_freq,fov_pe) (m),
        num_pe (# phase encodes), num_fe (# readout points),
        slice_locs : list of slice center locations (m)
        thk : slice thickness (m),
        gmax = max. gradient (T/m)
        srmax = max. slew rate (T/m/s)
        enc: encoding scheme - defines which of x, y, and z to use for freq/phase/slice select
            default: enc =(1,2,3); that is, x = 1 (freq. enc.), y = 2 (phase enc.), z = 3 (slice select.)


    Note: successful construction of sequence depends on gmax, srmax, RF, and slice selection parameters
          check printed timing values - if any is negative, then you might want to change gmax and srmax
          since pulseq files will be exclusively used starting from ver. 2, this limitation will not be fixed
    """
    def __init__(self,tr,te,flip_angle,fov,num_pe,num_fe,slice_locs,thk,gmax,srmax,enc=(1,2,3)):
        super().__init__(name="Gradient-recalled echo")
        self._tr = tr
        self._te = te
        self._fa = flip_angle
        self._fov = fov
        self._npe = num_pe # (number of phase encodes)
        self._nfe = num_fe # (number of readout points)
        self._slice_locs = slice_locs # location of slices (off center)
        self._ns = np.shape(slice_locs)[0] # number of slices
        self._thk = thk # slice thickness
        self._enc = enc # encoding. 1: fe; 2: pe; 3: ss; default is x-fe y-pe z-ss
        self._gmax = gmax
        self._srmax = srmax
        self._events = []

        rtmin = self._gmax/self._srmax

        # Calculate parameters & shapes and store
        # rf: store as base pulse + offsets
        #     rf will be offset during simulation
        gre_pulse = blc.SincRFPulse(bandwidth=thk*self.GAMMA_BAR*self._gmax,
                                    offset=0, num_zeros=12, flip_angle=self._fa,
                                    raster_time=1e-6,init_phase=0)

        rf_dur = gre_pulse.get_duration()
        # Calculate min TE and use max(minTE,input TE) as actual TE
        dkf = 1/fov[0]  # dk in freq. enc. direction
        dkp = 1/fov[1]  # dk in phase enc. direction

        dt = dkf/(self.GAMMA_BAR*self._gmax) # dwell time
        ro_time = self._nfe*dt
        rw_time = (ro_time - rtmin)/2

        if (self.GAMMA_BAR*self._gmax*rtmin) > (dkp*self._npe/2):
            pe_time = 0
            pe_amplitude = self._gmax * (dkp*self._npe/2)/(self.GAMMA_BAR*self._gmax*rtmin)

        else:
            pe_time = (dkp*self._npe/2)/(self.GAMMA_BAR*self._gmax) - rtmin
            pe_amplitude = self._gmax

        ssrf_time = (rf_dur - rtmin)/2
        # Gradient info
        # slice-selecting gradient
        g_ss_amp = self._gmax * (np.array(enc) == 3)
        g_ss = blc.TrapzGradient(rise_time=rtmin, flat_time=rf_dur, fall_time=rtmin, amplitude=g_ss_amp)

        # slice refocusing gradient
        g_ssrf_amp = -self._gmax * (np.array(enc) == 3)
        g_ssrf = blc.TrapzGradient(rise_time=rtmin, flat_time=ssrf_time, fall_time=rtmin, amplitude=g_ssrf_amp)

        # phase-encoding gradient & scaling
        g_pe_amp_max = pe_amplitude * (np.array(enc) == 2)
        g_pe = blc.TrapzGradient(rise_time=rtmin, flat_time=pe_time, fall_time=rtmin, amplitude=g_pe_amp_max)
        pe_scales = np.arange(-1,1,2/self._npe)

        # frequency encoding gradient -  re-winder
        g_rw_amp = -self._gmax * (np.array(enc) == 1)
        g_rw = blc.TrapzGradient(rise_time=rtmin, flat_time=rw_time, fall_time=rtmin, amplitude=g_rw_amp)

        # frequency encoding gradient - readout
        g_ro_amp = self._gmax * (np.array(enc) == 1)
        g_ro = blc.TrapzGradient(rise_time=rtmin, flat_time=ro_time, fall_time=rtmin, amplitude=g_ro_amp)

        # Define gradient library
        self._grad_types = [1, 2, 3]
        self._grad_dict = {1: [g_ss], 2: [g_ssrf, g_pe, g_rw], 3: [g_ro]}
        self._grad_scales = {1: [[1]],
                             2: [[1], pe_scales, [1]],
                             3: [[1]]}
        ###########################################################################################################
        print("rf duration is %.2E" % rf_dur)
        print("rt_min is %.2E" % rtmin)
        print("ro_time is %.2E" % ro_time)
        print("rw_time is %.2E" % rw_time)
        print("ssrf_time is %.2E" % ssrf_time)
        print("pe_time is %.2E" % pe_time)
        ###########################################################################################################

        min_te = rf_dur/2 + max(pe_time,rw_time,ssrf_time) + ro_time/2 + 4*rtmin
        if self._te < min_te:
            self._te = min_te
            print("TE is shorter than min TE; using min TE instead")
        print("Using TE = %.1f ms" % (1e3*self._te))
        te_delay = self._te - min_te
        min_tr = (self._te + ro_time/2 + 2*rtmin + rf_dur/2)
        if self._tr < min_tr:
            raise ValueError("TR is too short for given TE ! Please redo.")
        tr_delay = self._tr - min_tr

        # RF info
        self._rf_types = [1]
        self._rf_offsets = np.array(self._slice_locs)*self.GAMMA_BAR*self._gmax
        self._rf_dict = {1: gre_pulse}


        # ADCs
        self._adc = blc.ADC(num_samples=self._nfe,dwell_time=dt,delay=0)

        # Delays
        self._delay_types = [1,2]
        self._delay_dict = {1: te_delay, 2: tr_delay}

        # Make sequence
        for u in range(self._ns):  # for each slice
            for v in range(self._npe):  # for each phase encode
                self._events.append("rf")
                self._events.append("grad")
                self._events.append("delay") # TE delay
                self._events.append("readout")
                self._events.append("delay") # TR delay

    def get_rf(self,rf_ind):
        """
        Returns indexed rf pulse; used by simulator
            rf is offset depending on slice location
        Note: after rf_ind hits the end, it wraps back around
        """
        n = np.shape(self._rf_types)[0]
        m = np.shape(self._rf_offsets)[0]
        rf_type = self._rf_types[rf_ind%n]
        this_rf = self._rf_dict[rf_type]
        new_rf = copy.deepcopy(this_rf)
        new_rf.add_offset(self._rf_offsets[rf_ind % m])
        return new_rf

    def get_grads(self,grad_ind):
        """
        Returns indexed list of gradients [g1, g2, g3,...]
        Used by simulator
        Note: index wraps around; dict of gradient index -> list of scales is used for scaling phase encoding gradients
        """
        n = np.shape(self._grad_types)[0]
        grad_type = self._grad_types[grad_ind % n]
        these_grads = self._grad_dict[grad_type]
        m = np.shape(these_grads)[0]
        output_grads = []

        for q in range(m): # for each gradient listed in the current type
            if q == 1:
                scale_ind = int((grad_ind - 1)/3) % np.shape(self._grad_scales[grad_type][q])[0]
            else:
                scale_ind = 0
            sc = self._grad_scales[grad_type][q][scale_ind]
            output_grads.append(blc.get_scaled_gradient(these_grads[q],s=sc))

        return output_grads

    def get_delay(self,delay_ind):
        """
        Returns indexed delay (a single value in unit of seconds)
        Used by simulator
        """
        n = np.shape(self._delay_types)[0]
        delay_type = self._delay_types[delay_ind % n]
        return self._delay_dict[delay_type]

    def get_adc(self):
        return self._adc

    def get_signal_dims(self):
        return self._nfe, self._npe, self._ns

    def get_enc(self):
        return self._enc



