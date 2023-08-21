#!/usr/bin/env python3
#
# To run a single test, use e.g.:
# python -m unittest test_server.ServerTest.test_bad_packet

import socket, time, unittest
import numpy as np
import matplotlib.pyplot as plt
import warnings

import pdb
st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz, grad_board
from server_comms import *

class ServerTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    def setUp(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port))
        self.packet_idx = 0

    def tearDown(self):
        self.s.close()

    def test_version(self):
        full_test = True  # test a range of different versions; otherwise just the current one
        debug_replies = False
        def diff_equal(client_ver):
            return {'errors': ['not all client commands were understood']}

        def diff_info(client_ver):
            return {'infos': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' differs slightly from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug)],
             'errors': ['not all client commands were understood']}

        def diff_warning(client_ver):
            return {'warnings': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' different from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug)],
             'errors': ['not all client commands were understood']}

        def diff_error(client_ver):
            return {'errors': ['Client version {:d}.{:d}.{:d}'.format(*client_ver) +
                              ' significantly different from server version {:d}.{:d}.{:d}'.format(
                                  version_major, version_minor, version_debug),
                               'not all client commands were understood']}

        if full_test:
            versions = [ (1,0,1), (1,0,5), (1,3,100), (1,3,255), (2,5,7), (255,255,255) ]
            expected_outcomes = [diff_info, diff_equal, diff_warning, diff_warning, diff_error, diff_error]
        else:
            versions = [ (1,0,1) ]
            expected_outcomes = [diff_info]


        for v, ee in zip(versions, expected_outcomes):
            # send an unknown command to make sure system handles it gracefully
            packet = construct_packet({'asdfasdf':1}, self.packet_idx, version=v)
            reply = send_packet(packet, self.s)
            expected_reply = [reply_pkt, 1, 0, version_full, {'UNKNOWN1': -1}, ee(v)]
            if debug_replies:
                print("Reply         : ", reply)
                print("Expected reply: ", expected_reply)
                if reply == expected_reply:
                    print("Equal")
                else:
                    print("Not equal! Debugging...")
                    st()
            self.assertEqual(reply, expected_reply)

    def test_idle(self):
        """ Make sure the server state is (or becomes) idle, all the RX and TX buffers are empty, etc."""
        real = send_packet(construct_packet({'are_you_real':0}, self.packet_idx), self.s)[4]['are_you_real']
        if real == "hardware" or real == "simulation":
            buf_empties = 0xffffff
        elif real == "software":
            buf_empties = 0

        packet = construct_packet({'regstatus': 0})

        def check_status():
            """Check status of the firmware, returning True if idle,
            and always returning the last-read ADC value"""
            reply = send_packet(packet, self.s)
            registers = reply[4]['regstatus']

            # registers
            exec_reg = registers[0]
            status_reg = registers[1]
            status_latch_reg = registers[2]

            # status fields
            fhdo_busy = 0x20000
            ocra1_busy = 0x10000
            fhdo_adc = 0xffff

            # status latch fields
            fhdo_err = 0x4
            ocra1_err = 0x2
            ocra1_data_lost = 0x1

            if (status_latch_reg & fhdo_err) or (status_latch_reg & ocra1_err) or (status_latch_reg & ocra1_data_lost):
                warnings.warn("Gradient error occurred during test_idle! Might have been caused by another of the tests.")

            adc_value = status_reg & fhdo_adc
            idle = True
            if (status_reg & fhdo_busy) or (status_reg & ocra1_busy):
                idle = False

            return idle, adc_value

        for k in range(1000):
            idle, adc_value = check_status()
            if idle:
                break

        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full, {'regstatus': [0, adc_value, 0, 0, 0, buf_empties, 0]}, {}])

    def test_bad_packet(self):
        packet = construct_packet([1,2,3])
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {},
                          {'errors': ['no commands present or incorrectly formatted request']}])

    def test_bus(self):
        print_speeds = False
        real = send_packet(construct_packet({'are_you_real':0}, self.packet_idx), self.s)[4]['are_you_real']
        if real == "hardware":
            deltas = (0.2, 2, 2)
            times = (1.5, 131.0, 158.5) # numerical operation, bus read, bus write on hardware
            loops = 1000000
        elif real == "simulation":
            deltas = (0.1, 5000, 3500)
            times = (0.05, 10430, 5000)
            loops = 1000
        elif real == "software":
            deltas = (3, 5, 5)
            times = (5, 7, 7)
            loops = 10000000

        packet = construct_packet({'test_bus':loops}, self.packet_idx)
        reply = send_packet(packet, self.s)
        null_t, read_t, write_t = reply[4]['test_bus']

        loops_norm = loops/1e6
        if print_speeds:
            print(f"{real} data: null_t: {null_t/loops:.2f}, read_t: {read_t/loops:.2f}, write_t: {write_t/loops:.2f} us / cycle")
        if real == "hardware":
            self.assertAlmostEqual(null_t/1e3, times[0] * loops_norm, delta = deltas[0] * loops_norm) # 1 flop takes ~1.5 ns on average
            self.assertAlmostEqual(read_t/1e3, times[1] * loops_norm, delta = deltas[1] * loops_norm) # 1 read takes ~141.9 ns on average
            self.assertAlmostEqual(write_t/1e3, times[2] * loops_norm, delta = deltas[2] * loops_norm) # 1 write takes ~157.9 ns on average
        elif real == "simulation":
            # Might not be true if you're on a slow computer, but should be fine for most post-2015 PCs
            self.assertLess(null_t/loops, 1.0)
            self.assertLess(read_t/loops, 100.0)
            self.assertLess(write_t/loops, 100.0)

    @unittest.skip("marga devel")
    def test_net(self):
        real = send_packet(construct_packet({'are_you_real':0}, self.packet_idx), self.s)[4]['are_you_real']
        if real == "hardware":
            loops = [10, 1000, 100000]
            times = (1.5, 131.0, 158.5) # upper-bound times for network transfers
        elif real == "simulation":
            loops = [10, 1000, 100000]
            times = (1.5, 131.0, 158.5) # upper-bound times for network transfers
        elif real == "software":
            loops = [10, 1000, 100000]
            times = (1.5, 131.0, 158.5) # upper-bound times for network transfers
        packet = construct_packet({'test_net':10}, self.packet_idx)
        # VN: continue here

    def test_fpga_clk(self):
        packet = construct_packet({'fpga_clk': [0xdf0d, 0x03f03f30, 0x00100700]})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply, [reply_pkt, 1, 0, version_full, {'fpga_clk': 0}, {}])

    def test_fpga_clk_partial(self):
        packet = construct_packet({'fpga_clk': [0xdf0d,  0x03f03f30]})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'fpga_clk': -1},
                          {'errors': ["you only provided some FPGA clock control words; check you're providing all 3"]}]
        )

    @unittest.skip("marga devel")
    def test_several_okay(self):
        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 10,
                                   'rx_div': 250,
                                   'tx_size': 32767,
                                   'raw_tx_data': b"0000000000000000"*4096,
                                   'grad_div': (303, 32),
                                   'grad_ser': 1,
                                   'grad_mem': b"0000"*8192,
                                   'acq_rlim':10000,
                                   })
        reply = send_packet(packet, self.s)

        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': 0, 'rx_div': 0,
                           'tx_size': 0, 'raw_tx_data': 0, 'grad_div': 0, 'grad_ser': 0,
                           'grad_mem': 0, 'acq_rlim': 0},
                          {'infos': [
                              'tx data bytes copied: 65536',
                              'gradient mem data bytes copied: 32768']}]
        )

    @unittest.skip("marga devel")
    def test_several_some_bad(self):
        # first, send a normal packet to ensure everything's in a known state
        packetp = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                    'tx_div': 10, # 81.38ns sampling for 122.88 clock freq, 80ns for 125
                                    'rx_div': 250,
                                    'raw_tx_data': b"0000000000000000"*4096
        })
        send_packet(packetp, self.s)

        # Now, try sending with some issues
        packet = construct_packet({'lo_freq': 0x7000000, # floats instead of uints
                                   'tx_div': 100000,
                                   'rx_div': 32767,
                                   'tx_size': 65535,
                                   'raw_tx_data': b"0123456789abcdef"*4097,
                                   'grad_div': (1024, 0),
                                   'grad_ser': 16,
                                   'grad_mem': b"0000"*8193,
                                   'acq_rlim': 10,
                                   })

        reply = send_packet(packet, self.s)

        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'lo_freq': 0, 'tx_div': -1, 'rx_div': -1, 'tx_size': -1, 'raw_tx_data': -1, 'grad_div': -1, 'grad_ser': -1, 'grad_mem': -1, 'acq_rlim': -1},
                          {'errors': ['TX divider outside the range [1, 10000]; check your settings',
                                      'RX divider outside the range [25, 8192]; check your settings',
                                      'TX size outside the range [1, 32767]; check your settings',
                                      'too much raw TX data',
                                      'grad SPI clock divider outside the range [1, 63]; check your settings',
                                      'serialiser enables outside the range [0, 0xf], check your settings',
                                      'too much grad mem data: 32772 bytes > 32768',
                                      'acquisition retry limit outside the range [1000, 10,000,000]; check your settings'
                                      ]}
                          ])

    @unittest.skipUnless(grad_board == "gpa-fhdo", "requires GPA-FHDO board")
    def test_grad_adc(self):
        print_adc_reads = False
        # initialise SPI
        spi_div = 40
        upd = False # update on MSB writes
        send_packet(construct_packet( {'direct': 0x00000000 | (2 << 0) | (spi_div << 2) | (0 << 8) | (upd << 9)} ), self.s)

        # ADC defaults, same as in grad_board.GPAFHDO.init_hw()
        init_words = [
            0x0005000a, # DAC trigger reg, soft reset of the chip
            0x00030100, # DAC config reg, disable internal ref
            0x40850000, # ADC reset
            0x400b0600, 0x400d0600, 0x400f0600, 0x40110600, # input ranges for each ADC channel
            0x00088000, 0x00098000, 0x000a8000, 0x000b8000 # set each DAC channel to output 0
        ]

        real = send_packet(construct_packet({'are_you_real':0}, self.packet_idx), self.s)[4]['are_you_real']
        if real in ['simulation', 'software']:
            expected = [ 0 ] * ( len(init_words) - 1 )
        else:
            expected = [ 0xffff ] + [0x0600] * ( len(init_words) - 2)

        readback = []

        for iw in init_words:
            # direct commands to grad board; send MSBs then LSBs
            send_packet(construct_packet( {'direct': 0x02000000 | (iw >> 16)}), self.s)
            send_packet(construct_packet( {'direct': 0x01000000 | (iw & 0xffff)}), self.s)

            # read ADC each time

            # status reg = 5, ADC word is lower 16 bits
            adc_read = send_packet(construct_packet({'regrd': 5}), self.s)[4]['regrd']
            if print_adc_reads and adc_read != 0:
                print("ADC read: ", adc_read)
            time.sleep(0.01)
            readback.append( adc_read & 0xffff )
            # if readback != r:
            #     warnings.warn( "ADC data expected: 0x{:0x}, observed 0x{:0x}".format(w, readback) )

        self.assertEqual(expected, readback[1:]) # ignore 1st word, since it depends on the history of ADC transfers

    def test_leds(self):
        # This test is mainly for the simulator, but will alter hardware LEDs too
        for k in range(256):
            packet = construct_packet({'direct': 0x0f000000 + int((k & 0xff) << 8)})
            reply = send_packet(packet, self.s)
            self.assertEqual(reply,
                             [reply_pkt, 1, 0, version_full,
                              {'direct': 0}, {}])

        packet = construct_packet({'direct': 0x0f00a500}) # leds: a5
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'direct': 0}, {}])

        packet = construct_packet({'direct': 0x0f002400}) # leds: 24
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'direct': 0}, {}])

        # kill some time for the LEDs to change in simulation
        packet = construct_packet({'regstatus': 0})
        for k in range(2):
            reply = send_packet(packet, self.s)

    def test_mar_mem(self):
        mar_mem_bytes = 4 * 65536 # full memory
        # mar_mem_bytes = 4 * 2 # several writes for testing

        # everything should be fine
        raw_data = bytearray(mar_mem_bytes)
        for m in range(mar_mem_bytes):
            raw_data[m] = m & 0xff
        packet = construct_packet({'mar_mem' : raw_data})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'mar_mem': 0},
                          {'infos': ['mar mem data bytes copied: {:d}'.format(mar_mem_bytes)] }
                          ])

        # a bit too much data
        raw_data = bytearray(mar_mem_bytes + 1)
        for m in range(mar_mem_bytes):
            raw_data[m] = m & 0xff
        packet = construct_packet({'mar_mem' : raw_data})
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full,
                          {'mar_mem': -1},
                          {'errors': ['too much mar mem data: {:d} bytes > {:d} -- streaming not yet implemented'.format(mar_mem_bytes + 1, mar_mem_bytes)] }
                          ])

    @unittest.skip("marga devel")
    def test_acquire_simple(self):
        # For comprehensive tests, see test_loopback.py
        samples = 10
        packet = construct_packet({'acq': samples})
        reply = send_packet(packet, self.s)
        acquired_data_raw = reply[4]['acq']
        data = np.frombuffer(acquired_data_raw, np.complex64)

        self.assertEqual(reply[:4], [reply_pkt, 1, 0, version_full])
        self.assertEqual(len(acquired_data_raw), samples*8)
        self.assertIs(type(data), np.ndarray)
        self.assertEqual(data.size, samples)

        if False:
            plt.plot(np.abs(data));plt.show()

    @unittest.skip("rewrite needed")
    def test_bad_packet_format(self):
        packet = construct_packet({'configure_hw':
                                   {'lo_freq': 7.12345, # floats instead of uints
                                    'tx_div': 1.234}})
        reply_packet = send_packet(packet, self.s)
        # CONTINUE HERE: this should be handled gracefully by the server
        st()
        self.assertEqual(reply_packet,
                         [reply, 1, 0, version_full, {'configure_hw': 3}, {}]
        )

    @unittest.skip("comment this line out to shut down the server after testing")
    def test_exit(self): # last in alphabetical order
        packet = construct_packet( {}, 0, command=close_server_pkt)
        reply = send_packet(packet, self.s)
        self.assertEqual(reply,
                         [reply_pkt, 1, 0, version_full, {}, {'infos': ['Shutting down server.']}])

def throughput_test(s):
    packet_idx = 0

    for k in range(7):
        msg = msgpack.packb(construct_packet({'test_server_throughput': 10**k}))

        process(send_msg(msg, s))
        packet_idx += 2

def random_test(s):
    # Random other packet
    process(send_msg(msgpack.packb(construct_packet({'boo': 3}) , s)))

def shutdown_server(s):
    msg = msgpack.packb(construct_packet( {}, 0, command=close_server))
    process(send_msg(msg, s), print_all=True)

def test_client(s):
    packet_idx = 0
    pkt = construct_packet( {
        'configure_hw': {
            'fpga_clk_word1': 0x1,
            'fpga_clk_word2': 0x2
            # 'fpga_clk_word3': 0x3,
        },
    }, packet_idx)
    process(send_msg(msgpack.packb(pkt), s), print_all=True)

def main_test():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip_address, port))
        # throughput_test(s)
        test_client(s)
        # shutdown_server(s)

if __name__ == "__main__":
    # main_test()
    unittest.main()
