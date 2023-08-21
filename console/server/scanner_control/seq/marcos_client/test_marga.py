#!/usr/bin/env python3
# Basic hacks and tests for marga system
#

import numpy as np
import matplotlib.pyplot as plt

import socket, time
from local_config import ip_address, port, grad_board, fpga_clk_freq_MHz
from server_comms import *

from marmachine import *
import marcompile as fc

import pdb
st = pdb.set_trace

def get_exec_state(socket, display=True):
    reply, status = command({'regstatus': 0}, socket, print_infos=True)
    exec_reg = reply[4]['regstatus'][0]
    state = exec_reg >> 24
    pc = exec_reg & 0xffffff
    if display:
        print('state: {:d}, PC: {:d}'.format(state, pc))
    return state, pc

def run_manual_test(data, interval=0.001, timeout=20):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        # mem data
        # data = np.zeros(65536, dtype=np.uint32)
        command( {'mar_mem': data.tobytes()} , s)

        # Start execution
        command({'ctrl': 0x1}, s, print_infos=True)

        for k in range(int(timeout/interval)):
            time.sleep(interval)
            state, pc = get_exec_state(s)
            if (state == STATE_HALT):
                # one final acquisition to kill time on the simulation
                get_exec_state(s)
                break

        # Stop execution
        command({'ctrl': 0x2}, s, print_infos=True)

        # Flush RX data
        rx_data = command ({'flush_rx':0}, s, print_infos=True)[0][4]['flush_rx']

        # Auto-close server if it's simulating
        if command({'are_you_real':0}, s)[0][4]['are_you_real'] == "simulation":
            send_packet(construct_packet({}, 0, command=close_server_pkt), s)

        return rx_data

def run_streaming_test(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        packet_idx = 0

        if False: # clear memory first
            command( {'mar_mem': np.zeros(65536, dtype=np.uint32).tobytes()} , s)

        # mem data
        rx_data, msgs = command({'run_seq': data.tobytes()} , s)

        # print(msgs)

        # Auto-close server if it's simulating
        if command({'are_you_real':0}, s)[0][4]['are_you_real'] == "simulation":
            send_packet(construct_packet({}, 0, command=close_server_pkt), s)

        return rx_data

def leds():
    time_interval_s = 4 # how long the total sequence should run for
    total_states = 256
    delay = int(np.round(time_interval_s * fpga_clk_freq_MHz * 1e6 / total_states))

    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0
    for k in range(256):
        raw_data[addr] = insta(IWAIT, delay - 1); addr += 1 # -1 because the next instruction takes 1 cycle
        # raw_data[addr] = insta(IWAIT, 0x300000); addr += 1 # -1 because the next instruction takes 1 cycle
        raw_data[addr] = instb(GATES_LEDS, 0, (k & 0xff) << 8); addr += 1
    # go idle
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def marcompile_test():

    lc = fc.csv2bin("/tmp/mar_test1.csv")
    lc.append(insta(IFINISH, 0))
    raw_data = np.array(lc, dtype=np.uint32)
    print(raw_data.size)
    return raw_data

# Utility function to generate RX control words instead of doing it inline
def rx_ctrl(delay, rx0_en, rx1_en, rx0_rate_valid, rx1_rate_valid,
            rx0_lo, rx1_lo, rx0_rst_n=True, rx1_rst_n=True):
    return instb( RX_CTRL, delay,
                  (rx1_en << 9) | (rx0_en << 8) | (rx1_rst_n << 7) | (rx0_rst_n << 6) \
                  | (rx1_rate_valid << 5) | (rx0_rate_valid << 4) \
                  | (rx1_lo << 2) | (rx0_lo << 0) )

def example_tr_loop():
    lo_freq0 = 5 # MHz
    lo_freq1 = 6 # MHz
    lo_amp = 100 # percent
    raw_data = np.zeros(1000000, dtype=np.uint32)
    addr = 0

    cic0_decimation = 100
    cic1_decimation = 200
    dds_demod_ch = 3

    tr_loops = 2

    # Initially, turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in marga
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in marga
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, 0); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, 0x8000); addr += 1

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    # configure RX settings: both channels to use DDS source 0
    # reset CICs
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, 0, 0, 0, 0); addr += 1
    # take them out of reset
    raw_data[addr] = rx_ctrl(40, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch, 1, 1); addr += 1
    # set control buses to new rates (immediately for now)
    raw_data[addr] = instb(RX0_RATE, 0, cic0_decimation); addr += 1
    raw_data[addr] = instb(RX1_RATE, 0, cic1_decimation); addr += 1

    # briefly signal that there's a new rate
    raw_data[addr] = rx_ctrl(40, 0, 0, 1, 1, dds_demod_ch, dds_demod_ch); addr += 1
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1

    for k in range(tr_loops):

        ## Initial TX, hard pulse
        wait = 1.6276 # 1.6276us will lead to exactly 200 cycles

        initial_block_delay = 4

        # initial delay of 100 cycles, taking into account the next block
        raw_data[addr] = insta(IWAIT, 100 - initial_block_delay - 5); addr += 1

        addr_tx_start = addr
        raw_data[addr] = instb(GATES_LEDS, initial_block_delay + 1, 0x1); addr += 1 # turn on TX gate
        raw_data[addr] = instb(TX0_I, initial_block_delay - 1, 0xc000); addr += 1
        raw_data[addr] = instb(TX0_Q, initial_block_delay - 2, 0x3fff); addr += 1
        raw_data[addr] = instb(TX1_I, initial_block_delay - 3, 0xc000); addr += 1
        raw_data[addr] = instb(TX1_Q, initial_block_delay - 4, 0x3fff); addr += 1

        # take into account the time offset due to internal instructions and pre/post-wait instruction delays
        # Written assuming the TX and TX gate outputs should be perfectly in sync - in reality a delay is probably wise
        initial_block_delay = 4
        raw_data[addr] = insta(IWAIT, int(np.round(wait * fpga_clk_freq_MHz)) + 1 - initial_block_delay - addr + addr_tx_start); addr += 1
        raw_data[addr] = instb(GATES_LEDS, initial_block_delay + 1, 0x0); addr += 1; # turn off TX gate
        raw_data[addr] = instb(TX0_I, initial_block_delay - 1, 0x0); addr += 1
        raw_data[addr] = instb(TX0_Q, initial_block_delay - 2, 0x0); addr += 1
        raw_data[addr] = instb(TX1_I, initial_block_delay - 3, 0x0); addr += 1
        raw_data[addr] = instb(TX1_Q, initial_block_delay - 4, 0x0); addr += 1
        addr_tx_end = addr

        # Wait then acquire some data, wait some more, stop acquisition
        raw_data[addr] = insta(IWAIT, 100 - 4 - addr + addr_tx_end); addr += 1 # wait 100 cycles
        raw_data[addr] = rx_ctrl(40, 1, 1, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
        raw_data[addr] = insta(IWAIT, 500 - 5); addr += 1 # wait 500 cycles
        raw_data[addr] = rx_ctrl(40, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1

    # Go idle
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    print(addr)
    return raw_data

def tx_short():
    lo_freq0 = 5 # MHz
    lo_freq1 = 10
    lo_freq2 = 1.5
    lo_freq3 = 13.333333
    lo_amp = 100 # percent

    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0

    # Turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in marga
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in marga
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, 0); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, 0x8000); addr += 1

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 20); addr += 1 # max I value

    steps = 300
    pattern0 = np.hstack([ np.linspace(1, 0, steps//3), np.linspace(1, -1, steps//3), np.linspace(-1, 1, steps//3) ])
    pattern1 = np.hstack([ np.linspace(0, 1, steps//3), np.linspace(1, 0, steps//3), np.linspace(0, -1, steps//3) ])
    tdata0 = np.round(0x7fff * pattern0).astype(np.int16)
    tdata1 = np.round(0x7fff * pattern1).astype(np.int16)

    for k in range(steps):
        hv0 = tdata0[k] & 0xffff
        hv1 = tdata1[k] & 0xffff
        raw_data[addr] = instb(TX0_I, 3, hv0); addr += 1
        raw_data[addr] = instb(TX0_Q, 2, hv0); addr += 1
        raw_data[addr] = instb(TX1_I, 1, hv1); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, hv1); addr += 1

    # do mid-scale output, and change frequency
    raw_data[addr] = instb(TX0_I, 3, 0x4000); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x4000); addr += 1

    # wait for 3us; shortened delay by 4 cycles for the next instructions to be right on time
    raw_data[addr] = insta(IWAIT, int(3 * fpga_clk_freq_MHz) - 4); addr += 1

    dds2_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq2).astype(np.uint32) # 31b phase accumulator in marga
    dds3_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq3).astype(np.uint32) # 31b phase accumulator in marga

    # switch frequency on both channels simultaneously, no reset
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds2_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds2_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds3_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds3_phase_step >> 16); addr += 1

    raw_data[addr] = insta(IWAIT, int(4 * fpga_clk_freq_MHz)); addr += 1

    # go idle
    raw_data[addr] = instb(TX0_I, 3, 0); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x8000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x8000); addr += 1
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def rx_short():
    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0

    # turn DDS source 0 on
    lo_freq0 = 5 # MHz
    cic_decimation = 4
    dds_demod_ch = 0
    acquisition_ticks = 800
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in marga
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    raw_data[addr] = instb(DDS0_PHASE_LSB, 1, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, dds0_phase_step >> 16); addr += 1

    # configure RX settings: both channels to use DDS source 3 (DC), decimation of 10
    # reset CICs
    ## raw_data[addr] = instb(RX0_CTRL, 0, 0x0000); addr += 1
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, 0, 0); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    # take them out of reset later
    ## raw_data[addr] = instb(RX0_CTRL, 50, 0x8000 | cic_decimation | (dds_demod_ch << 12) ); addr += 1
    raw_data[addr] = rx_ctrl(50, 1, 0, 0, 0, dds_demod_ch, 0); addr += 1
    raw_data[addr] = instb(RX0_RATE, 0, cic_decimation); addr += 1
    # briefly signal that there's a new rate
    ## raw_data[addr] = instb(RX0_CTRL, 50, 0xc000 | cic_decimation | (dds_demod_ch << 12) ); addr += 1
    raw_data[addr] = rx_ctrl(50, 1, 0, 1, 0, dds_demod_ch, 0); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 49, 0xf000 | cic_decimation); addr += 1
    # end the new rate flag (the buffers are not empty so no offset time is needed)
    ## raw_data[addr] = instb(RX0_CTRL, 0, 0x8000 | cic_decimation | (dds_demod_ch << 12) ); addr += 1 # decimation may not be needed here
    raw_data[addr] = rx_ctrl(0, 1, 0, 0, 0, dds_demod_ch, 0); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0xb000 | cic_decimation); addr += 1

    # just acquire data for a while
    raw_data[addr] = insta(IWAIT, acquisition_ticks - 1); addr += 1

    # reset RX again
    ## raw_data[addr] = instb(RX0_CTRL, 1, 0x3000); addr += 1
    raw_data[addr] = rx_ctrl(1, 0, 0, 0, 0, dds_demod_ch, 0); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x3000); addr += 1

    # turn off DDS0
    raw_data[addr] = instb(DDS0_PHASE_LSB, 1, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, 0); addr += 1

    # go idle
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    return raw_data

def loopback(cic0_decimation=7, cic1_decimation=10):
    lo_freq0 = 4 # MHz
    lo_freq1 = 4
    lo_freq2 = 1.5
    lo_freq3 = 1.5
    lo_amp = 100 # percent
    dds_demod_ch = 0

    extra_time = 20

    raw_data = np.zeros(65536, dtype=np.uint32)
    addr = 0

    # Turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in marga
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in marga
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 7, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 6, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 5, 0); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 4, 0x8000); addr += 1

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 0, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 0, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    # configure RX settings: both channels to use DDS source 3 (DC), decimation of 10
    # reset CICs
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch, 0, 0); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    # take them out of reset later
    raw_data[addr] = rx_ctrl(40, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
    raw_data[addr] = instb(RX0_RATE, 0, cic0_decimation); addr += 1
    raw_data[addr] = instb(RX1_RATE, 0, cic1_decimation); addr += 1
    # briefly signal that there's a new rate
    raw_data[addr] = rx_ctrl(40, 0, 0, 1, 1, dds_demod_ch, dds_demod_ch); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 49, 0xf000 | cic_decimation); addr += 1
    # end the new rate flag and start acquisition (the buffers are not empty so no offset time is needed)
    raw_data[addr] = rx_ctrl(0, 1, 1, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1

    # Wait a bit, so that all buffers are emptied
    raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    if False:
        ## NOTE: example of how to send arbitrary data to RX0 or RX1 CIC:
        # first, set the output bus to the data you want to send (in this case, cic0_decimation)
        raw_data[addr] = instb(RX0_RATE, 0, cic0_decimation); addr += 1
        # next, strobe the 'bus 0 valid' line of the RX control buffer (no delays needed if it's not the Xilinx CIC)
        raw_data[addr] = rx_ctrl(0, 0, 0, 1, 0, dds_demod_ch, dds_demod_ch); addr += 1
        raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
        # same for channel 1:
        raw_data[addr] = instb(RX1_RATE, 0, cic1_decimation); addr += 1
        raw_data[addr] = rx_ctrl(0, 0, 0, 0, 1, dds_demod_ch, dds_demod_ch); addr += 1
        raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
        # wait 100 cycles and reset both CICs, then wait 100 more and take them out of reset and start acquisition
        raw_data[addr] = rx_ctrl(100, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch, 0, 0); addr += 1
        raw_data[addr] = rx_ctrl(100, 1, 1, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1

        # Wait a bit, so that all buffers are emptied
        raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    steps = 300
    pattern0 = np.hstack([ np.linspace(1, 0, steps//3), np.linspace(1, -1, steps//3), np.linspace(-1, 1, steps//3) ])
    # pattern1 = np.hstack([ np.linspace(0, 1, steps//3), np.linspace(1, 0, steps//3), np.linspace(0, -1, steps//3) ])
    pattern1 = pattern0
    tdata0 = np.round(0x7fff * pattern0).astype(np.int16)
    tdata1 = np.round(0x7fff * pattern1).astype(np.int16)

    for k in range(steps):
        hv0 = tdata0[k] & 0xffff
        hv1 = tdata1[k] & 0xffff
        raw_data[addr] = instb(TX0_I, 3, hv0); addr += 1
        raw_data[addr] = instb(TX0_Q, 2, hv0); addr += 1
        raw_data[addr] = instb(TX1_I, 1, hv1); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, hv1); addr += 1

    # do mid-scale output, and change frequency
    raw_data[addr] = instb(TX0_I, 3, 0x4000); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x4000); addr += 1

    # wait for 3us; shortened delay by 4 cycles for the next instructions to be right on time
    raw_data[addr] = insta(IWAIT, int(3 * fpga_clk_freq_MHz) - 4); addr += 1

    dds2_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq2).astype(np.uint32) # 31b phase accumulator in marga
    dds3_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq3).astype(np.uint32) # 31b phase accumulator in marga

    # switch frequency on both channels simultaneously, no reset
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds2_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds2_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds3_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds3_phase_step >> 16); addr += 1

    # wait for extra acquisition time
    raw_data[addr] = insta(IWAIT, int(extra_time * fpga_clk_freq_MHz)); addr += 1

    # end RX
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1

    # go idle
    raw_data[addr] = instb(TX0_I, 3, 0); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x0000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x0000); addr += 1
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    # raw_data = raw_data[:addr] # truncate
    return raw_data

def long_loopback():
    lo_freq0 = 4 # MHz
    lo_freq1 = 4.01
    lo_freq2 = 1.5
    lo_freq3 = 1.51
    lo_amp = 100 # percent

    # These settings lead to a fairly normal sequence
    if False:
        cic0_decimation = 1000
        cic1_decimation = 500
        extra_time = 20
        sine_ts = 2000
        max_addr = 400000

    # These settings lead to memory-buffer-low events
    if False:
        cic0_decimation = 50
        cic1_decimation = 94
        extra_time = 20
        sine_ts = 5
        max_addr = 141000

    # These settings lead to a very long sequence, with a few memory buffer low warnings
    if False:
        cic0_decimation = 100
        cic1_decimation = 200
        extra_time = 20
        sine_ts = 10
        max_addr = 500000

    dds_demod_ch = 0
    raw_data = np.zeros(max_addr, dtype=np.uint32) # massive sequence
    addr = 0

    # Turn on LO
    dds0_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq0).astype(np.uint32) # 31b phase accumulator in marga
    dds1_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq1).astype(np.uint32) # 31b phase accumulator in marga
    assert dds0_phase_step < 2**31, "DDS frequency outside of valid range"
    assert dds1_phase_step < 2**31, "DDS frequency outside of valid range"

    # zero the phase increment initially and reset the phase
    raw_data[addr] = instb(DDS0_PHASE_LSB, 7, 0); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 6, 0x8000); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 5, 0); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 4, 0x8000); addr += 1

    # set the phase increment, start phase going
    raw_data[addr] = instb(DDS0_PHASE_LSB, 0, dds0_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 0, dds0_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 0, dds1_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds1_phase_step >> 16); addr += 1

    # allow the new phase to propagate through the chain before setting nonzero I/Q values
    raw_data[addr] = insta(IWAIT, 100); addr += 1 # max I value

    # configure RX settings: both channels to use DDS source 3 (DC), decimation of 10
    # reset CICs
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0x0000); addr += 1
    # take them out of reset later
    raw_data[addr] = rx_ctrl(40, 1, 1, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
    raw_data[addr] = instb(RX0_RATE, 0, cic0_decimation); addr += 1
    raw_data[addr] = instb(RX1_RATE, 0, cic1_decimation); addr += 1
    # briefly signal that there's a new rate
    raw_data[addr] = rx_ctrl(40, 1, 1, 1, 1, dds_demod_ch, dds_demod_ch); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 49, 0xf000 | cic_decimation); addr += 1
    # end the new rate flag (the buffers are not empty so no offset time is needed)
    raw_data[addr] = rx_ctrl(0, 1, 1, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1
    # raw_data[addr] = instb(RX1_CTRL, 0, 0xb000 | cic_decimation); addr += 1

    steps = 300
    pattern0 = np.hstack([ np.linspace(1, 0, steps//3), np.linspace(1, -1, steps//3), np.linspace(-1, 1, steps//3) ])
    # pattern1 = np.hstack([ np.linspace(0, 1, steps//3), np.linspace(1, 0, steps//3), np.linspace(0, -1, steps//3) ])
    pattern1 = pattern0
    tdata0 = np.round(0x7fff * pattern0).astype(np.int16)
    tdata1 = np.round(0x7fff * pattern1).astype(np.int16)

    for k in range(steps):
        hv0 = tdata0[k] & 0xffff
        hv1 = tdata1[k] & 0xffff
        raw_data[addr] = instb(TX0_I, 3, hv0); addr += 1
        raw_data[addr] = instb(TX0_Q, 2, hv0); addr += 1
        raw_data[addr] = instb(TX1_I, 1, hv1); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, hv1); addr += 1

    # do mid-scale output, and change frequency
    raw_data[addr] = instb(TX0_I, 3, 0x4000); addr += 1
    raw_data[addr] = instb(TX0_Q, 2, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_I, 1, 0x4000); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x4000); addr += 1

    # wait for 3us; shortened delay by 4 cycles for the next instructions to be right on time
    raw_data[addr] = insta(IWAIT, int(3 * fpga_clk_freq_MHz) - 4); addr += 1

    dds2_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq2).astype(np.uint32) # 31b phase accumulator in marga
    dds3_phase_step = np.round(2**31 / fpga_clk_freq_MHz * lo_freq3).astype(np.uint32) # 31b phase accumulator in marga

    # switch frequency on both channels simultaneously, no reset
    raw_data[addr] = instb(DDS0_PHASE_LSB, 3, dds2_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS0_PHASE_MSB, 2, dds2_phase_step >> 16); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_LSB, 1, dds3_phase_step & 0xffff); addr += 1
    raw_data[addr] = instb(DDS1_PHASE_MSB, 0, dds3_phase_step >> 16); addr += 1

    # wait for extra acquisition time
    raw_data[addr] = insta(IWAIT, int(extra_time * fpga_clk_freq_MHz)); addr += 1

    ### Long sinusoid of slowly increasing frequency
    start_addr = addr
    end_addr = max_addr - 1000 # reserve 1000 spots at the end for final commands
    phase_points = (end_addr - start_addr) // 5
    t = np.linspace(0, 1, phase_points)
    c = np.cos(2*np.pi*20*t) * (0.2 + 0.5*t)
    s = np.sin(2*np.pi*13*t) * (0.1 + 0.8*t)
    cb = np.round(c * 0x7fff).astype(np.int16) & 0xffff
    sb = np.round(s * 0x7fff).astype(np.int16) & 0xffff

    for k in range(phase_points):
        raw_data[addr] = instb(TX0_I, 0, cb[k]); addr += 1
        raw_data[addr] = instb(TX0_Q, 0, sb[k]); addr += 1
        raw_data[addr] = instb(TX1_I, 0, cb[k]); addr += 1
        raw_data[addr] = instb(TX1_Q, 0, sb[k]); addr += 1
        raw_data[addr] = insta(IWAIT, sine_ts); addr += 1

    # end RX
    raw_data[addr] = rx_ctrl(0, 0, 0, 0, 0, dds_demod_ch, dds_demod_ch); addr += 1

    # go idle
    raw_data[addr] = instb(TX0_I, 0, 0); addr += 1
    raw_data[addr] = instb(TX0_Q, 0, 0x8000); addr += 1
    raw_data[addr] = instb(TX1_I, 0, 0); addr += 1
    raw_data[addr] = instb(TX1_Q, 0, 0x8000); addr += 1
    raw_data[addr] = insta(IFINISH, 0); addr += 1
    raw_data = raw_data[:addr] # truncate
    print(addr)
    return raw_data

if __name__ == "__main__":
    if False: # flash the LEDs
        run_streaming_test(leds())

    if False:
        data = long_loopback()
        st()

    if False: # single shot, plot RX
        res = run_streaming_test(example_tr_loop())
        # res = run_streaming_test(marcompile_test())

        rxd = res[4]['run_seq']
        # offsets = 1e8
        offsets = 0
        rx0_i = np.array(rxd['rx0_i']).astype(np.int32)
        rx0_q = np.array(rxd['rx0_q']).astype(np.int32)+offsets
        rx1_i = np.array(rxd['rx1_i']).astype(np.int32)+2*offsets
        rx1_q = np.array(rxd['rx1_q']).astype(np.int32)+3*offsets
        plt.plot(rx0_i)
        plt.plot(rx0_q)
        plt.plot(rx1_i)
        plt.plot(rx1_q)

        plt.legend(['rx0_i', 'rx0_q', 'rx1_i', 'rx1_q'])

        plt.show()

    if True: # single shot loopback

        # res = run_streaming_test(long_loopback())
        res = run_streaming_test(loopback())

        rxd = res[4]['run_seq']
        # offsets = 1e8
        offsets = 0
        rx0_i = np.array(rxd['rx0_i']).astype(np.int32)
        rx0_q = np.array(rxd['rx0_q']).astype(np.int32)
        rx1_i = np.array(rxd['rx1_i']).astype(np.int32)
        rx1_q = np.array(rxd['rx1_q']).astype(np.int32)
        plt.plot(rx0_i)
        plt.plot(rx0_q)
        plt.plot(rx1_i)
        plt.plot(rx1_q)

        plt.legend(['rx0_i', 'rx0_q', 'rx1_i', 'rx1_q'])

        plt.show()

    if False: # multiple trials, verify mean against saved
        verify_means = True
        save_means = False # If true, will overwrite reference data - use with care!

        trials = 100

        cic0_decimation = 6
        cic1_decimation = 11

        res = run_streaming_test(loopback(cic0_decimation, cic1_decimation))
        rxd = res[4]['run_seq']
        rx0_pts = len(rxd['rx0_i'])
        rx1_pts = len(rxd['rx1_i'])

        rx0_i = np.zeros([trials, rx0_pts]).astype(np.int32)
        rx0_q = np.zeros([trials, rx0_pts]).astype(np.int32)
        rx1_i = np.zeros([trials, rx1_pts]).astype(np.int32)
        rx1_q = np.zeros([trials, rx1_pts]).astype(np.int32)

        for k in range(trials):
            res = run_streaming_test(loopback(cic0_decimation, cic1_decimation))
            rxd = res[4]['run_seq']
            # offsets = 1e8
            offsets = 0
            rx0_i[k,:] = np.array(rxd['rx0_i']).astype(np.int32)
            rx0_q[k,:] = np.array(rxd['rx0_q']).astype(np.int32)
            rx1_i[k,:] = np.array(rxd['rx1_i']).astype(np.int32)
            rx1_q[k,:] = np.array(rxd['rx1_q']).astype(np.int32)

        # plt.legend(['rx0_i', 'rx0_q', 'rx1_i', 'rx1_q'])
        rx0_x = np.arange(rx0_pts) * cic0_decimation / fpga_clk_freq_MHz
        rx1_x = np.arange(rx1_pts) * cic1_decimation / fpga_clk_freq_MHz

        props = {'alpha': 0.3}

        plt.fill_between(rx0_x, rx0_i.min(0), rx0_i.max(0), **props)
        plt.fill_between(rx0_x, rx0_q.min(0), rx0_q.max(0), **props)
        plt.fill_between(rx1_x, rx1_i.min(0), rx1_i.max(0), **props)
        plt.fill_between(rx1_x, rx1_q.min(0), rx1_q.max(0), **props)

        rx0im, rx0qm, rx1im, rx1qm = rx0_i.mean(0), rx0_q.mean(0), rx1_i.mean(0), rx1_q.mean(0)
        rx0rms, rx1rms = np.sqrt(rx0im**2 + rx0qm**2), np.sqrt(rx1im**2 + rx1qm**2)

        plt.gca().set_prop_cycle(None)

        plt.plot(rx0_x, rx0im)
        plt.plot(rx0_x, rx0qm)
        plt.plot(rx1_x, rx1im)
        plt.plot(rx1_x, rx1qm)

        plt.plot(rx0_x, rx0rms)
        plt.plot(rx1_x, rx1rms)

        plt.xlabel('t (us)')
        plt.legend(['rx0_i', 'rx0_q', 'rx1_i', 'rx1_q', 'rx0_rms', 'rx1_rms'])

        if False:
            plt.plot(rx0_i.std(0))
            plt.plot(rx0_q.std(0))
            plt.plot(rx1_i.std(0))
            plt.plot(rx1_q.std(0))

        ## Verify data
        if verify_means:
            ref_data = np.load('ref_loopback.npz')
            rx0rmsr, rx1rmsr = ref_data['rx0rms'], ref_data['rx1rms']
            # rx0im[10] *= 5 # deliberately stimulate error
            for rx, rxr in zip([rx0rms, rx1rms], [rx0rmsr, rx1rmsr]):
                diff = ((rx/rx.max() - rxr/rxr.max())**2).sum() / rx.size
                if diff > 1e-5:
                    warnings.warn("Normalised loopback data is not as expected!")

        if save_means:
            np.savez_compressed('ref_loopback.npz', rx0rms=rx0rms, rx1rms=rx1rms)

        plt.show()

    if False:
        trials = 1
        for k in range(trials):
            rxd = run_manual_test(loopback(), interval=0.01, timeout=20)
            rx0 = np.array(rxd['ch0']).astype(np.int32)
            rx1 = np.array(rxd['ch1']).astype(np.int32)
            plt.plot(rx0[::2])
            plt.plot(rx0[1::2])
            plt.plot(rx1[::2])
            plt.plot(rx1[1::2])

        plt.xlabel('sample')
        plt.title('loopback, {:d} trials'.format(trials))
        plt.show()
