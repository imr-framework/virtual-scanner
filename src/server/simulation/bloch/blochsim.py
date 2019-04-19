"""
Bloch simulator class :D
Gehua Tong, March 2019

"""
import phantom as pht
import bloch as blc
import bloch_sequence as blcs
from pulseq.core.Sequence.sequence import Sequence as pulSequence
import numpy as np
import multiprocessing as mp

class BlochSimulator:
    """
    Simulator class
    Note that the sequence attribute is a Sequence object from either bloch.py or pulseq
    """
    def __init__(self,phantom,sequence):
        self._simulated = False
        self._recon_fin = False
        self._seq = sequence  # might be bloch.Sequence or pulseq.Sequence
        self._phantom = phantom
        self._signal = []
        self._images = []
        self._signal_store = []

        # Things that may be added
        # self._b0map (hardware-caused b0 imperfections)
        # self._Txb1map
        # self._Rxb1map

    def get_signal(self):
        return self._signal

    # For parallel processing

    def apply_ps_to(self,loc_ind,freq_offset=0):
        signal = []
        location = self._phantom.get_location(loc_ind)
        isc = blc.SpinGroup(loc=location, params=self._phantom.get_params(loc_ind), df=freq_offset)
        rf_ind = 0
        grad_ind = 0
        delay_ind = 0
        signal_ind = 0

        for event in self._seq.get_events():
            # 1. rf: with optional gradient (no relaxation)
            if event == "rf":
                isc.apply_rf(pulse=self._seq.get_rf(rf_ind),
                             grad=self._seq.get_grads(grad_ind)[0],
                             trapz_opt=True)
                rf_ind += 1
                grad_ind += 1

            # 2. grad: relaxation with gradient
            elif event == "grad":
                this_grad = self._seq.get_grads(grad_ind)
                isc.fpwg(grads=this_grad)
                grad_ind += 1

            # 3. delay: relaxation
            elif event == "delay":
                this_delay = self._seq.get_delay(delay_ind)
                isc.delay(t=this_delay)
                delay_ind += 1

            # 4. readout: gradient with ADC readout
            elif event == "readout":
                this_grad = self._seq.get_grads(grad_ind)
                s = isc.readout(this_grad[0], adc=self._seq.get_adc())
                grad_ind += 1
                # Store signal in appropriate place
                signal.append(s)
                signal_ind += 1
    ###
        return signal


    def simulate(self):
        if isinstance(self._seq,blcs.BlochSequence):
            self.simulate_blcseq()
            self._simulated = True
        elif isinstance(self._seq,pulSequence):
            self.simulate_pulseq()
            self._simulated = True



    def simulate_blcseq(self):
        """ MAIN SIMULATION METHOD
        Run simulation for a Sequence object from bloch.py
        """
        ph = self._phantom
        ph_shape = ph.get_shape()
        ph_fov = ph.get_fov()
        vsize = ph.get_vsize()

        # Define coordinates
        xs = np.arange(-ph_fov[0]/2+vsize/2, ph_fov[0]/2, vsize)
        ys = np.arange(-ph_fov[1]/2+vsize/2, ph_fov[1]/2, vsize)
        zs = np.arange(-ph_fov[2]/2+vsize/2, ph_fov[2]/2, vsize)

        # Now: only 2D cartesian supported
        nfe,npe,ns = self._seq.get_signal_dims()
        self._signal = np.zeros((npe*ns,nfe),dtype=np.complex_)

        # For each point in the phantom
        for u in range(ph_shape[0]):
            for v in range(ph_shape[1]):
                for w in range(ph_shape[2]):
                    # For each spin group at that point
                    Df = [0]
                    for q in range(1):  # refine this later as part of the phantom class
                        isc = blc.SpinGroup(loc=(xs[u], ys[v], zs[w]), params=ph.get_params((u, v, w)), df=Df[q])
                        # Apply each step of the sequence in sequence's event list
                        rf_ind = 0
                        grad_ind = 0
                        delay_ind = 0
                        signal_ind = 0
                        for event in self._seq.get_events():
                            # 1. rf: with optional gradient (no relaxation)
                            if event == "rf":
                                isc.apply_rf(pulse=self._seq.get_rf(rf_ind),
                                             grad=self._seq.get_grads(grad_ind)[0],
                                             trapz_opt=True)
                                rf_ind += 1
                                grad_ind += 1

                            # 2. grad: relaxation with gradient
                            elif event == "grad":
                                this_grad = self._seq.get_grads(grad_ind)
                                isc.fpwg(grads=this_grad)
                                grad_ind += 1

                            # 3. delay: relaxation
                            elif event == "delay":
                                this_delay = self._seq.get_delay(delay_ind)
                                isc.delay(t=this_delay)
                                delay_ind += 1

                            # 4. readout: gradient with ADC readout
                            elif event == "readout":
                                this_grad = self._seq.get_grads(grad_ind)
                                s = isc.readout(this_grad[0],adc=self._seq.get_adc())
                                grad_ind += 1
                                # Store signal in appropriate place
                                self._signal[signal_ind] += s
                                signal_ind += 1

        print("Simulation complete!")

    def simulate_pulseq(self):
        return []

    def cartesian_recon(self):  # simple recon for testing simulator
        if self._simulated:
            # Do cartesian recon. here
            print(0)
            self._recon_fin = True
            # recon
            #self._images
        else:
            print("Not simulated yet!")

    def show_image(self):
        return 0


# Helper methods
# Take x,y,z coordinates and make into list of 3-tuples of all possible combinations
def get_list_locs(Xs,Ys,Zs):
    list_locs = []
    for x in Xs:
        for y in Ys:
            for z in Zs:
                list_locs.append((x,y,z))

    return list_locs
