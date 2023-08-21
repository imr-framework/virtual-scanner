#!/usr/bin/env python3
#
# Classes to handle GPA initialisation, calibration and communication
#
# They need to at least implement the following methods:
#
# init_hw() to program the GPA chips on power-up or reset them if
# they're in an undefined state
#
# write_dac() to send binary numbers directly to a DAC (the method
# should take care of bit shifts, extra bits etc - the user supplies
# only the binary DAC output code)
#
# read_adc() to retrieve a binary ADC word from the GPA (if enabled);
# should output the binary ADC code, but shouldn't be responsible for
# off-by-one codes (i.e. for the GPA-FHDO, it doesn't have to correct
# the ADC's behaviour of sending the previously-read voltage with the
# current transfer)
#
# calibrate() to prepare the data for user-defined calibration
# procedures (such as scaling/offset, piecewise interpolation, etc) to
# take place later. This might be outputting test currents and
# measuring the actual current using an ADC, loading a file containing
# manually acquired code-vs-voltage calibration data, etc.
# Once the method has been run, the system should be ready to handle:
#
# float2bin() to convert a list of input Numpy arrays in units of the
# full-scale DAC output (i.e. [-1, 1]) into the binary BRAM data to
# reproduce the multi-channel waveform on the GPA - should apply any
# desired calibrations/transforms internally.
#
# bin2float() to convert the binary data into [-1, 1] floats.
#
# keys() returns the gradient board-specific labels
#
# key_convert() to convert from the user-facing dictionary key labels
# to gradient board-specific labels, and also return a channel
#
# TODO: actually use class inheritance here, instead of two separate classes

import numpy as np
from numpy.polynomial import Polynomial
import time, warnings
import matplotlib.pyplot as plt
import console.server.scanner_control.seq.marcos_client.local_config as lc

import pdb
st = pdb.set_trace

grad_clk_t = 1/lc.fpga_clk_freq_MHz # ~8.14ns period for RP-122

class OCRA1:
    def __init__(self,
                 server_command_f,
                 max_update_rate=0.1):
        """ max_update_rate is in MSPS for updates on a single channel; used to choose the SPI clock divider """

        spi_cycles_per_tx = 30 # actually 24, but including some overhead
        self.spi_div = int(np.floor(1 / (spi_cycles_per_tx * max_update_rate * grad_clk_t))) - 1
        if self.spi_div > 63:
            self.spi_div = 63 # max value, < 100 ksps

        # bind function from Experiment class, or replace with something else for debugging
        self.server_command = server_command_f

        # Default calibration settings for all channels: linear transformation for now
        self.cal_values = [ (1,0), (1,0), (1,0), (1,0) ]

        self.bin_config = {
            'initial_bufs': np.array([
                # see marga.sv, gradient control lines (lines 186-190, 05.02.2021)
                # strobe for both LSB and LSB, reset_n = 1, spi div as given, grad board select (1 = ocra1, 2 = gpa-fhdo)
                (1 << 9) | (1 << 8) | (self.spi_div << 2) | 1,
                0, 0,
                0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0], dtype=np.uint16),
            'latencies': np.array([
                0, 268, 268, # grad latencies match SPI div
                0, 0, # rx
                0, 0, 0, 0, # tx
                0, 0, 0, 0, 0, 0, # lo phase
                0, 0 # gates and LEDs
            ], dtype=np.uint16)}

    def wait_for_ocra1_iface_idle(self):
        """try multiple times until ocra1_iface core is idle -- read marga
         register 5 and look at the ocra1 busy bit
        """
        for m in range(2): # waste a few cycles initially (mostly important for simulation)
            rd, _ = self.server_command({'regrd': 5})

        trials = 1000
        while trials > 0:
            rd, _ = self.server_command({'regrd': 5})
            ocra1_busy = rd[4]['regrd'] & 0x10000
            # print("trial {:d} busy {:d}".format(1001 - trials, ocra1_busy))
            if ocra1_busy == 0:
                # break
                trials = 0
            else:
                trials -= 1 # try again

    def init_hw(self):
        init_words = [
            # lower 24 bits sent to ocra1, upper 8 bits used to control ocra1 serialiser channel + broadcast
            0x00400004, 0x02400004, 0x04400004, 0x07400004, # reset DACs to power-on values
            0x00200002, 0x02200002, 0x04200002, 0x07200002, # set internal amplifier
            0x00100000, 0x02100000, 0x04100000, 0x07100000, # set outputs to 0
        ]

        # configure main grad ctrl word first, in particular switch it to update the serialiser strobe only in response to LSB changes;
        # strobe the reset of the core just in case
        # (marga buffer address = 0, 8 MSBs of the 32-bit word)
        self.server_command({'direct': 0x00000000 | (1 << 0) | (self.spi_div << 2) | (0 << 8) | (0 << 9)})
        self.server_command({'direct': 0x00000000 | (1 << 0) | (self.spi_div << 2) | (1 << 8) | (0 << 9)})

        for k, iw in enumerate(init_words):
            self.wait_for_ocra1_iface_idle()

            # direct commands to grad board; send MSBs then LSBs
            self.server_command({'direct': 0x02000000 | (iw >> 16)})
            self.server_command({'direct': 0x01000000 | (iw & 0xffff)})

        # restore main grad ctrl word to its default value, and reset the OCRA1 iface core
        self.server_command({'direct': 0x00000000})

    def write_dac(self, channel, value, gated_writes=True):
        """gated_writes: if the caller knows that marga will already be set
        to send data to the serialiser only on LSB updates, this can
        be set to False. However if it's incorrectly set to False,
        there may be spurious writes to the serialiser in direct mode
        as the MSBs and LSBs are output by the buffers at different
        times (for a timed marga sequence, the buffers either update
        simultaneously or only a single one updates at a time to save
        instructions).
        """
        assert 0, "Not yet written, sorry!"

    def read_adc(self, channel, value):
        assert 0, "OCRA1 has no ADC!"

    def calibrate(self):
        # Fill more in here
        st()

    def keys(self):
        return ["ocra1_" + l for l in ["vx", "vy", "vz", "vz2"] ]

    def key_convert(self, user_key):
        # convert key from user-facing dictionary to marcompile format
        vstr = user_key.split('_')[1]
        ch_list = ['vx', 'vy', 'vz', 'vz2']
        return "ocra1_" + vstr, ch_list.index(vstr)

    def float2bin(self, grad_data, channel=0):
        cv = self.cal_values[channel]
        gd_cal = grad_data * cv[0] + cv[1] # calibration
        return np.round(131071.49 * gd_cal).astype(np.uint32) & 0x3ffff # 2's complement

    def bin2float(self, grad_bin):
        return ( ((grad_bin & 0x3ffff).astype(np.int32) ^ (1 << 17)) - (1 << 17) ).astype(np.int32) / 131072

class GPAFHDO:
    def __init__(self,
                 server_command_f,
                 max_update_rate=0.1):
        """ max_update_rate is in MSPS for updates on a single channel; used to choose the SPI clock divider """
        fhdo_max_update_rate = max_update_rate * 4 # single-channel serial, so needs to be faster

        spi_cycles_per_tx = 30 # actually 24, but including some overhead
        self.spi_div = int(np.floor(1 / (spi_cycles_per_tx * fhdo_max_update_rate * grad_clk_t))) - 1
        if self.spi_div > 63:
            self.spi_div = 63 # max value, < 100 ksps

        self.adc_spi_div = 30 # slow down when ADC transfers are being done

        # bind function from Experiment class, or replace with something else for debugging
        self.server_command = server_command_f

        # TODO: will this ever need modification?
        self.grad_channels = 4

        # try to get from local_config.py
        try:
            self.gpa_current_per_volt = lc.gpa_fhdo_current_per_volt
        except AttributeError:
            self.gpa_current_per_volt = 2.5 # if it doesn't match your grad board, add to your local_config.py

        # initialize gpa fhdo calibration with ideal values
        # self.dac_values = np.array([0x7000, 0x8000, 0x9000])
        # self.gpaCalValues = np.ones((self.grad_channels,self.dac_values.size))
        # self.dac_values = np.array([0x0, 0xffff])
        # self.gpaCalValues = np.tile(self.expected_adc_code_from_dac_code(self.dac_values), (self.grad_channels, 1))

        self.gpaCal = []
        for k in range(self.grad_channels):
            self.gpaCal.append( Polynomial([0, 1]) ) # polynomials for calibration

        self.bin_config = {
            'initial_bufs': np.array([
                # see marga.sv, gradient control lines (lines 186-190, 05.02.2021)
                # strobe for both MSB and LSB, reset_n = 1, spi div = 10, grad board select (1 = ocra1, 2 = gpa-fhdo)
                (1 << 9) | (1 << 8) | (self.spi_div << 2) | 2,
                0, 0,
                0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0], dtype=np.uint16),
            'latencies': np.array([
                0, 276, 276, # grad latencies match SPI div
                0, 0, # rx
                0, 0, 0, 0, # tx
                0, 0, 0, 0, 0, 0, # lo phase
                0, 0 # gates and LEDs
            ], dtype=np.uint16)}

    def wait_for_gpa_fhdo_iface_idle(self):
        """try multiple times until gpa_fhdo_iface core is idle -- read marga
         register 5 and look at the fhdo busy bit
        """
        for m in range(2): # waste a few cycles initially (mostly important for simulation)
            rd, _ = self.server_command({'regrd': 5})

        trials = 1000
        while trials > 0:
            rd, _ = self.server_command({'regrd': 5})
            fhdo_busy = rd[4]['regrd'] & 0x20000
            # print("trial {:d} busy {:d}".format(1001 - trials, fhdo_busy))
            if fhdo_busy == 0:
                # break
                trials = 0
            else:
                trials -= 1 # try again

    def init_hw(self):
        # write defaults
        init_words = [
            0x0005000a, # DAC trigger reg, soft reset of the chip
            0x00030100, # DAC config reg, disable internal ref
            0x40850000, # ADC reset
            0x400b0600, 0x400d0600, 0x400f0600, 0x40110600, # input ranges for each ADC channel
            0x00088000, 0x00098000, 0x000a8000, 0x000b8000 # set each DAC channel to output 0
        ]

        # configure main grad ctrl word first, in particular switch it to update the serialiser strobe only in response to LSB changes;
        # gpa_fhdo_iface core has no reset, so no need to strobe it unlike for ocra1
        # (marga buffer address = 0, 8 MSBs of the 32-bit word)

        self.server_command({'direct': 0x00000000 | (2 << 0) | (self.adc_spi_div << 2) | (0 << 8) | (0 << 9)})

        for iw in init_words:
            self.wait_for_gpa_fhdo_iface_idle()

            # direct commands to grad board; send MSBs then LSBs
            self.server_command({'direct': 0x02000000 | (iw >> 16)})
            self.server_command({'direct': 0x01000000 | (iw & 0xffff)})

        # restore main grad ctrl word to default
        self.server_command({'direct': 0x00000000})

    def update_on_msb_writes(self, upd, spi_div=None):
        if spi_div is None:
            spi_div = self.spi_div
        """upd: bool; set to True for default mode, False when direct writes
        are being done. As a side effect, always sets the SPI divisor."""
        self.server_command({'direct': 0x00000000 | (2 << 0) | (spi_div << 2) | (0 << 8) | (upd << 9)})

    def write_dac(self, channel, value, gated_writes=True):
        """gated_writes: if the caller knows that marga will already be set
        to send data to the serialiser only on LSB updates, this can
        be set to False to avoid unnecessary server commands. However
        if it's incorrectly set to False, there may be spurious writes
        to the serialiser, because in direct DAC write mode (which
        this function uses) the MSBs and LSBs are output by the
        buffers at different times (for a timed marga sequence, the
        buffers either update simultaneously or only a single one
        updates at a time to save instructions) and a spurious update
        with only one 16b block changed could be sent. Leave it on
        True if unsure.
        """
        if gated_writes:
            self.update_on_msb_writes(False)

        msbs = 0x0008 | channel | (channel << 9) # extra channel word for gpa_fhdo_iface, not sure if it's currently used
        value_msbs = value >> 16
        if (value_msbs != 0) and (value_msbs != msbs):
            warnings.warn("You requested channel {:d}, which does not match that of the binary word provided (0x{:0x}). Are you using channels consistently?".format(channel, value))

        self.server_command({'direct': 0x02000000 | msbs }) # MSBs
        self.server_command({'direct': 0x01000000 | int(value) & 0xffff }) # LSBs

        # restore main grad ctrl word to respond to LSB or MSB changes
        if gated_writes:
            self.update_on_msb_writes(True)

    def read_adc(self, channel, gated_writes=True):
        """ see write_dac docstring
        Assumes SPI divisor and DAC/ADC settings have already been initialised through init_hw() at some point"""
        if gated_writes:
            self.update_on_msb_writes(False, self.adc_spi_div)

        adc_word = 0x40c00000 | (channel << 18)
        self.server_command({'direct': 0x02000000 | (adc_word >> 16) }) # MSBs
        self.server_command({'direct': 0x01000000 }) # LSBs
        rd, _ = self.server_command({'regrd': 5})

        # restore main grad ctrl word to respond to LSB or MSB changes
        if gated_writes:
            self.update_on_msb_writes(True, self.spi_div)
        return rd[4]['regrd'] & 0xffff # lower 16 bits of marga reg 5

    def expected_adc_code_from_dac_code_old(self, dac_code):
        """
        a helper function for calibrate_gpa_fhdo(). It calculates the expected adc value for a given dac value if every component was ideal.
        The dac codes that pulseq generates should be based on this ideal assumption. Imperfections will be automatically corrected by calibration.
        """
        dac_voltage = 5 * (dac_code / 0xffff)
        v_ref = 2.5
        gpa_current = (dac_voltage-v_ref) * self.gpa_current_per_volt
        r_shunt = 0.2
        adc_voltage = gpa_current*r_shunt+v_ref
        adc_gain = 4.096*1.25   # ADC range register setting has to match this
        adc_code = adc_voltage/adc_gain * 0xffff
        #print('DAC code {:d}, DAC voltage {:f}, GPA current {:f}, ADC voltage {:f}, ADC code {:d}'.format(dac_code,dac_voltage,gpa_current,adc_voltage,adc_code))
        return adc_code

    def grad2adc(self, grad_vals):
        """ Calculate expected ideal ADC code for a given grad value ( i.e. [-1, 1]) as input """
        v_ref = 2.5
        gpa_current = self.grad2amp(grad_vals)
        r_shunt = 0.2
        adc_voltage = gpa_current*r_shunt+v_ref
        adc_gain = 4.096*1.25   # ADC range register setting has to match this
        adc_code = adc_voltage/adc_gain * 0xffff
        #print('DAC code {:d}, DAC voltage {:f}, GPA current {:f}, ADC voltage {:f}, ADC code {:d}'.format(dac_code,dac_voltage,gpa_current,adc_voltage,adc_code))
        return adc_code

    def adc2grad(self, adc_val):
        """ Calculate expected grad value to produce an observed ADC code """
        v_ref = 2.5
        r_shunt = 0.2
        adc_gain = 4.096*1.25   # ADC range register setting has to match this
        adc_voltage = adc_val * adc_gain / 0xffff
        gpa_current = (adc_voltage - v_ref) / r_shunt
        return self.amp2grad(gpa_current)

    def calculate_corrected_dac_code_old(self,channel,dac_code):
        """
        calculates the correction factor for a given dac code by doing linear interpolation on the data points collected during calibration
        """
        return np.round( np.interp(self.expected_adc_code_from_dac_code(dac_code), self.gpaCalValues[channel], self.dac_values) ).astype(np.uint32)

    def amp2grad(self, ampere):
        """ Calculate ideal [-1,1] gradient value required for a particular current output """
        v_ref = 2.5 # nominal midpoint of DAC output
        v_max = 5 # maximum DAC output voltage
        # dac_code = np.round( (ampere / self.gpa_current_per_volt + v_ref)/5 * 0xffff ).astype(int)
        # return dac_code
        return ampere / ( self.gpa_current_per_volt * (v_max - v_ref) ) # full-scale grad val is +/- 1

    def grad2amp(self, grad_vals):
        """ Reverse of amp2grad() """
        v_ref = 2.5 # nominal midpoint of DAC output
        v_max = 5 # maximum DAC output voltage
        return grad_vals * ( self.gpa_current_per_volt * (v_max - v_ref) ) # full-scale grad val is +/- 1

    def calibrate(self,
                  channels=[0,1,2,3],
                  max_current=5,
                  num_calibration_points=20,
                  # gpa_current_per_volt=3.75,
                  averages=5,
                  settle_time=0.001, # ms after each write
                  poly_degree=3, # cubic by default, can go higher/lower if desired
                  test_cal=False, # Purely for debugging
                  plot=False):

        for chan in channels:
            grad_vals = np.linspace(self.amp2grad(-max_current),
                                    self.amp2grad(max_current),
                                    num_calibration_points )
            dac_vals = self.float2bin(grad_vals, channel=chan, cal=test_cal)
            adc_vals = np.zeros_like(grad_vals)

            for k, dv in enumerate(dac_vals):
                self.write_dac(chan, dv)
                # time.sleep(0.001) # 1ms
                self.read_adc(chan) # dummy
                time.sleep(0.001) # 1ms
                for m in range(averages):
                    adc_vals[k] += self.read_adc(chan) # real

            # restore to rough midpoint, in case calibration fails
            self.write_dac(chan, self.float2bin(0, chan, cal=False) )

            # expected_adc_vals = self.grad2adc(grad_vals)
            adc_vals /= averages # normalise again

            observed_grad_vals = self.adc2grad(adc_vals)

            if test_cal:
                # just plot residuals of data
                plt.plot(grad_vals, observed_grad_vals - grad_vals, label='Residuals')
                plt.xlabel('Grad vals (normalised, [-1, 1])')
                plt.ylabel('Residuals (observed - expected grad vals) (normalised, [-1, 1])')
                plt.show()
            else:

                # perform polynomial fit
                try:
                    p = Polynomial.fit(observed_grad_vals, grad_vals, poly_degree)
                except ValueError:
                    warnings.warn("Poly fit failed due to numerical problems -- perhaps no current could be output.")

                # check that first normalised coefficient is close to 1
                coeff = 2*p.coef[1]/np.abs(p.domain).sum()
                if coeff > 1.05 or coeff < 0.95:
                    warnings.warn("Poly slope coefficient {:f} for chan {:d} is outside [0.95, 1.05]; will not be used. Make sure the coils are connected to the GPA-FHDO and the system is correctly powered.".format(coeff, chan))
                else:
                    self.gpaCal[chan] = p

            self.write_dac(chan, self.float2bin(0, chan, cal=True) ) # restore to precise midpoint

    def apply_cal(self, grad_vals, chan):
        return self.gpaCal[chan](grad_vals)

    ## VN: commenting out the old calibration routine for now - can re-introduce it later
    def calibrate_old(self,
                  max_current = 2,
                  num_calibration_points = 10,
                  gpa_current_per_volt = 3.75,
                  averages=4,
                  settle_time=0.001, # ms after each write
                  plot=False):
        """
        performs a calibration of the gpa fhdo for every channel. The number of interpolation points in self.dac_values can
        be adapted to the accuracy needed.
        """
        self.update_on_msb_writes(True)

        self.gpa_current_per_volt = gpa_current_per_volt
        self.dac_values = np.round(np.linspace(self.ampere_to_dac_code(-max_current),self.ampere_to_dac_code(max_current),num_calibration_points))
        self.dac_values = self.dac_values.astype(int)
        self.gpaCalValues = np.ones((self.grad_channels,self.dac_values.size))
        for channel in range(self.grad_channels):
            if False:
                np.random.shuffle(self.dac_values) # to ensure randomised acquisition
            adc_values = np.zeros([self.dac_values.size, averages]).astype(np.uint32)
            gpaCalRatios = np.zeros(self.dac_values.size)
            for k, dv in enumerate(self.dac_values):
                self.write_dac(channel,dv, False)
                time.sleep(settle_time) # wait 1ms to settle

                self.read_adc(channel) # dummy read
                for m in range(averages):
                    adc_values[k][m] = self.read_adc(channel)
                self.gpaCalValues[channel][k] = adc_values.sum(1)[k]/averages
                gpaCalRatios[k] = self.gpaCalValues[channel][k]/self.expected_adc_code_from_dac_code(dv)
                #print('Received ADC code {:d} -> expected ADC code {:d}'.format(int(adc_values.sum(1)[k]/averages),self.expected_adc_code_from_dac_code(dv)))
            self.write_dac(channel,0x8000, False) # set gradient current back to 0

            if np.amax(gpaCalRatios) > 1.01 or np.amin(gpaCalRatios) < 0.99:
                print('Calibration for channel {:d} seems to be incorrect. Calibration factor is {:f}. Make sure a gradient coil is connected and gpa_current_per_volt value is correct.'.format(channel,np.amax(gpaCalRatios)))
            if plot:
                plt.plot(self.dac_values, adc_values.min(1), 'y.')
                plt.plot(self.dac_values, adc_values.max(1), 'y.')
                plt.plot(self.dac_values, adc_values.sum(1)/averages, 'b.')
                plt.xlabel('DAC word'); plt.ylabel('ADC word, {:d} averages'.format(averages))
                plt.grid(True)
                plt.show()

        # housekeeping
        self.update_on_msb_writes(True)

    def keys(self):
        return ["fhdo_" + l for l in ["vx", "vy", "vz", "vz2"] ]

    def key_convert(self, user_key):
        # convert key from user-facing dictionary to marcompile format
        vstr = user_key.split('_')[1]
        ch_list = ['vx', 'vy', 'vz', 'vz2']
        return "fhdo_" + vstr, ch_list.index(vstr)

    def float2bin(self, grad_vals, channel=0, cal=False):
        # cal: apply calibration or not
        # Not 2's complement - 0x0 word is ~0V (-10A), 0xffff is ~+5V (+10A)
        # gr_dacbits_cal = self.calculate_corrected_dac_code(channel,gr_dacbits)
        if cal:
            grad_vals_cal = self.apply_cal(grad_vals, channel)
        else:
            grad_vals_cal = grad_vals
        gr_dacbits = np.round(32767.51 * (grad_vals_cal + 1)).astype(np.uint16)
        gr = gr_dacbits | 0x80000 | (channel << 16)

        # # always broadcast for the final channel (TODO: probably not needed for GPA-FHDO, check then remove)
        # broadcast = channel == self.grad_channels - 1
        # grad_bram_data[channel::self.grad_channels] = gr | (channel << 25) | (broadcast << 24) # interleave data
        return gr | (channel << 25) # extra channel word for gpa_fhdo_iface, not sure if it's currently used

    def bin2float(self, grad_bin):
        return (grad_bin & 0xffff).astype(np.uint16) / 32768 - 1
