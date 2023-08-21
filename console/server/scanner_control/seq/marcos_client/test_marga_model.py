#!/usr/bin/env python3
#
# Integrated test of the MaRCoS server, HDL and compiler.
#
# Script compiles and runs the MaRCoS Verilator simulation (server +
# HDL model) and sends it various binaries generated using the MaRCoS
# client compiler, then compares the simulated hardware output against
# the expected output.
#
# To run a single test, use e.g.:
# python -m unittest test_marga_model.Modeltest.test_many_quick

from test_base import *

class ModelTest(unittest.TestCase):
    """Main test class for general HDL and compiler development/debugging;
    inputs to the API and UUT are either a CSV file or a dictionary,
    output is another CSV file, and comparison is either between the
    input and output CSV files (with allowance made for memory/startup
    delays) or between the output CSV and a reference CSV file. If a
    dictionary is used as input, a reference CSV file must be created.
    """

    @classmethod
    def setUpClass(cls):
        # TODO make this check for a file first
        subprocess.call(["make", "-j4", "-s", "-C", os.path.join(marga_sim_path, "build")])
        subprocess.call(["fallocate", "-l", "516KiB", "/tmp/marcos_server_mem"])
        subprocess.call(["killall", "marga_sim"], stderr=subprocess.DEVNULL) # in case other instances were started earlier

        warnings.simplefilter("ignore", mc.MarServerWarning)

    def setUp(self):
        # start simulation
        if marga_sim_fst_dump:
            self.p = subprocess.Popen([os.path.join(marga_sim_path, "build", "marga_sim"), "both", marga_sim_csv, marga_sim_fst],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT)
        else:
            self.p = subprocess.Popen([os.path.join(marga_sim_path, "build", "marga_sim"), "csv", marga_sim_csv],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT)


        # open socket
        time.sleep(0.05) # give marga_sim time to start up

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip_address, port)) # only connect to local simulator
        self.packet_idx = 0

    def tearDown(self):
        # self.p.terminate() # if not already terminated
        # self.p.kill() # if not already terminated
        self.s.close()

        if marga_sim_fst_dump:
            # open GTKWave
            os.system("gtkwave " + marga_sim_fst + " " + os.path.join(marga_sim_path, "src", "marga_sim.sav"))

    ## Tests are approximately in order of complexity

    def test_single(self):
        """ Basic state change on a single buffer """
        refl, siml = compare_csv("test_single", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_four_par(self):
        """ State change on four buffers in parallel """
        refl, siml = compare_csv("test_four_par", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_long_time(self):
        """ State change on four buffers in parallel """
        max_orig = mc.COUNTER_MAX
        mc.COUNTER_MAX = 0xfff # temporarily reduce max time used by compiler
        refl, siml = compare_csv("test_long_time", self.s, self.p)
        mc.COUNTER_MAX = max_orig
        self.assertEqual(refl, siml)

    def test_single_quick(self):
        """ Quick successive state changes on a single buffer 1 cycle apart """
        refl, siml = compare_csv("test_single_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_single_delays(self):
        """ State changes on a single buffer with various delays in between"""
        refl, siml = compare_csv("test_single_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_quick(self):
        """ Quick successive state changes on two buffers, one cycle apart """
        refl, siml = compare_csv("test_two_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_delays(self):
        """ State changes on two buffers, various delays in between """
        refl, siml = compare_csv("test_two_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_three_quick(self):
        """ Quick successive state changes on two buffers, one cycle apart """
        refl, siml = compare_csv("test_three_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_three_delays(self):
        """ Successive state changes on three buffers, two cycles apart """
        refl, siml = compare_csv("test_three_delays", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_mult_quick(self):
        """ Quick successive state changes on multiple buffers, 1 cycle apart """
        refl, siml = compare_csv("test_mult_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_many_quick(self):
        """ Many quick successive state changes on multiple buffers, all 1 cycle apart """
        refl, siml = compare_csv("test_many_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_stream_quick(self):
        """ Bursts of state changes on multiple buffers with uneven gaps for each individual buffer, each state change 1 cycle apart """
        refl, siml = compare_csv("test_stream_quick", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_times(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
        refl, siml = compare_csv("test_uneven_times", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_sparse(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart """
        refl, siml = compare_csv("test_uneven_sparse", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_cfg(self):
        """ Configuration and LED bits/words """
        refl, siml = compare_csv("test_cfg", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_rx_simple(self):
        """ RX window with realistic RX rate configuration, resetting and gating """
        refl, siml = compare_csv("test_rx_simple", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_two_uneven_latencies(self):
        """Simultaneous state changes on two buffers, however 2nd buffer is
        specified to have 1 cycle of latency more than 1st - so its changes
        occur earlier to compensate"""
        refl, siml = compare_csv("test_two_uneven_latencies", self.s, self.p,
                                  latencies=np.array([
                                      0,0,0,0,
                                      0,0,1,0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      0], dtype=np.uint16),
                                  self_ref=False)
        self.assertEqual(refl, siml)

    def test_many_uneven_latencies(self):
        """Simultaneous state changes on four buffers, however they are
        assumed to all have different latencies relative to each other - thus
        out-of-sync change requests turn out in sync"""
        refl, siml = compare_csv("test_many_uneven_latencies", self.s, self.p,
                                  latencies=np.array([
                                      0, 0, 0, # grad
                                      0, 0, # rx
                                      2, 4, 6, 8, # tx
                                      0, 0, 0, 0, 0, 0, # lo phase
                                      0, 0 # gates and LEDs, RX config
                                  ], dtype=np.uint16),
                                  self_ref=False)
        self.assertEqual(refl, siml)

    def test_fhd_single(self):
        """Single state change on GPA-FHDO x gradient output, default SPI
        clock divisor; simultaneous change on TX0i"""
        set_grad_board("gpa-fhdo")
        refl, siml = compare_csv("test_fhd_single", self.s, self.p, **fhd_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_fhd_series(self):
        """Series of state changes on GPA-FHDO x gradient output, default SPI
        clock divisor """
        set_grad_board("gpa-fhdo")
        refl, siml = compare_csv("test_fhd_series", self.s, self.p, **fhd_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_fhd_multiple(self):
        """A few state changes on GPA-FHDO gradient outputs, default SPI clock divisor"""
        set_grad_board("gpa-fhdo")
        refl, siml = compare_csv("test_fhd_multiple", self.s, self.p, **fhd_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_fhd_many(self):
        """Many state changes on GPA-FHDO gradient outputs, default SPI clock divisor - simultaneous with similar TX changes"""
        set_grad_board("gpa-fhdo")
        refl, siml = compare_csv("test_fhd_many", self.s, self.p, **fhd_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_fhd_too_fast(self):
        """Two state changes on GPA-FHDO gradient outputs, default SPI clock
        divisor - too fast for the divider, second state change won't
        be applied and server should notice the error
        """
        set_grad_board("gpa-fhdo")

        # Run twice, to catch two different warnings (I couldn't find a more straightforward way to do this)
        with self.assertWarns( RuntimeWarning, msg="expected gpa-fhdo error not observed") as cmr:
            refl, siml = compare_csv("test_fhd_too_fast", self.s, self.p, self_ref=False, **fhd_config)
        with self.assertWarns( UserWarning, msg="expected marcompile warning not observed") as cmu:
            self.tearDown()
            self.setUp()
            refl, siml = compare_csv("test_fhd_too_fast", self.s, self.p, self_ref=False, **fhd_config)

        restore_grad_board()
        # self.assertEqual( str(cm.exception) , "gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual( str(cmu.warning), "Gradient updates are too frequent for selected SPI divider. Missed samples are likely!")
        self.assertEqual( str(cmr.warning) , "SERVER ERROR: gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual(refl, siml)

    def test_oc1_single(self):
        """Single state change on ocra1 x gradient output, default SPI clock
        divisor; simultaneous change on TX0i
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_single", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_series(self):
        """Series of state changes on ocra1 x gradient output, default SPI
        clock divisor
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_series", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_two(self):
        """Two sets of simultaneous state changes on ocra1 gradient outputs,
        default SPI clock divisor
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_two", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_two_same(self):
        """One set of simultaneous identical state changes on ocra1 gradient
        outputs, then a single change later; default SPI clock divisor
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_two_same", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_four(self):
        """Four simultaneous state changes on ocra1 gradient outputs, default
        SPI clock divisor
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_four", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_series_same(self):
        """Several sets of simultaneous state changes on ocra1 gradient
        outputs, with MSBs and LSBs kept similar in a pattern to test
        a bug; default SPI clock divisor
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_series_same", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)
        # self.assertEqual(1, 0)

    def test_oc1_many(self):
        """Multiple simultaneous state changes on ocra1 gradient outputs, default
        SPI clock divisor
        """
        set_grad_board("ocra1")
        refl, siml = compare_csv("test_oc1_many", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_too_fast(self):
        """Two state changes on ocra1 gradient outputs, default SPI clock
        divisor - too fast for the divider, second state change won't
        be applied and server should notice the error
        """
        set_grad_board("ocra1")

        # Run twice, to catch two different warnings (I couldn't find a more straightforward way to do this)
        with self.assertWarns( RuntimeWarning, msg="expected ocra1 error not observed") as cmr:
            refl, siml = compare_csv("test_oc1_too_fast", self.s, self.p, self_ref=False, **oc1_config)
        with self.assertWarns( UserWarning, msg="expected marcompile warning not observed") as cmu:
            self.tearDown()
            self.setUp()
            refl, siml = compare_csv("test_oc1_too_fast", self.s, self.p, self_ref=False, **oc1_config)
            restore_grad_board()
        # self.assertEqual( str(cm.exception) , "gpa-fhdo gradient error; possibly missing samples")
        self.assertEqual( str(cmu.warning), "Gradient updates are too frequent for selected SPI divider. Missed samples are likely!")
        self.assertEqual( str(cmr.warning) , "SERVER ERROR: ocra1 gradient error; possibly missing samples")
        self.assertEqual(refl, siml)

    def test_single_dict(self):
        """ Basic state change on a single buffer. Dict version"""
        d = {'tx0_i': (np.array([100]), np.array([10000]))}
        refl, siml = compare_dict(d, "test_single", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_removed_instructions_dict(self):
        """Basic state change on a single buffer, with many more specified
        events than actual changes, leading to removed instructions and a
        series of warnings. Dict version"""
        reps = 2000
        d = {'tx0_i': ( np.arange(100, 100 + reps) , np.array([10000]*reps) )}
        with self.assertWarns( mc.MarRemovedInstructionWarning,
                               msg="expected marcompile warning not observed") as cmu:
            refl, siml = compare_dict(d, "test_single", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_four_par_dict(self):
        """ State change on four buffers in parallel. Dict version"""
        d = {'tx0_i': (np.array([100]), np.array([5000])),
             'tx0_q': (np.array([100]), np.array([15000])),
             'tx1_i': (np.array([100]), np.array([25000])),
             'tx1_q': (np.array([100]), np.array([35000]))}
        refl, siml = compare_dict(d, "test_four_par", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_uneven_sparse_dict(self):
        """ Bursts of state changes on multiple buffers with uneven gaps, each state change uneven numbers of cycles apart. Dict version"""
        d = {'tx0_i': (np.array([100, 110, 113, 114, 115, 130, 152, 153, 156, 170, 174, 300, 10000]),
                       np.array([1, 5, 9, 13, 17, 25, 33, 37, 41, 45, 49, 53, 57])),
             'tx0_q': (np.array([110, 113, 114, 115, 116, 152, 156, 170, 174, 300, 10001]),
                       np.array([6, 10, 14, 18, 22, 34, 42, 46, 50, 54, 62])),
             'tx1_i': (np.array([113, 114, 115, 116, 130, 152, 153, 170, 177, 300, 10001]),
                       np.array([11, 15, 19, 23, 27, 35, 39, 47, 51, 55, 63])),
             'tx1_q': (np.array([114, 115, 116, 130, 150, 152, 156, 170, 177, 300, 10000]),
                       np.array([16, 20, 24, 28, 32, 36, 44, 48, 52, 56, 60])) }
        refl, siml = compare_dict(d, "test_uneven_sparse", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_cfg_dict(self):
        """ Configuration and LED bits/words. Dict version"""
        d = {'rx0_rate': ( np.array([100, 110, 228, 230]), np.array([1, 0, 1, 0]) ),
             'rx1_rate': ( np.array([110, 120, 225, 228]), np.array([1, 0, 1, 0]) ),
             'rx0_rate_valid': ( np.array([120, 130, 222, 225]), np.array([1, 0, 1, 0]) ),
             'rx1_rate_valid': ( np.array([130, 140, 220, 222]), np.array([1, 0, 1, 0]) ),
             'rx0_rst_n': ( np.array([140, 150, 218, 220]), np.array([1, 0, 1, 0]) ),
             'rx1_rst_n': ( np.array([150, 160, 216, 218]), np.array([1, 0, 1, 0]) ),
             'rx0_en': ( np.array([160, 170, 215, 216]), np.array([1, 0, 1, 0]) ),
             'rx1_en': ( np.array([170, 180, 214, 215]), np.array([1, 0, 1, 0]) ),
             'tx_gate': ( np.array([180, 190, 213, 214]), np.array([1, 0, 1, 0]) ),
             'rx_gate': ( np.array([190, 200, 212, 213]), np.array([1, 0, 1, 0]) ),
             'trig_out': ( np.array([200, 210, 211, 212]), np.array([1, 0, 1, 0]) ),
             'leds': ( np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
                                 211, 212, 213, 214, 215, 216, 218, 220, 222, 225, 228, 230]),
                       np.array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11, 12,
                                   20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 255]) )
             }
        refl, siml = compare_dict(d, "test_cfg", self.s, self.p)
        self.assertEqual(refl, siml)

    def test_fhd_many_dict(self):
        """Many state changes on GPA-FHDO gradient outputs, default SPI clock divisor - simultaneous with similar TX changes. Dict version"""
        set_grad_board("gpa-fhdo")
        d = {'tx0_i': (np.array([600, 1800]), np.array([1, 65532])),
             'tx0_q': (np.array([900, 2100]), np.array([2, 65533])),
             'tx1_i': (np.array([1200, 2400]), np.array([3, 65534])),
             'tx1_q': (np.array([1500, 2700]), np.array([4, 65535])),
             'fhdo_vx': (np.array([600, 1800]), np.array([1, 65532])),
             'fhdo_vy': (np.array([900, 2100]), np.array([2, 65533])),
             'fhdo_vz': (np.array([1200, 2400]), np.array([3, 65534])),
             'fhdo_vz2': (np.array([1500, 2700]), np.array([4, 65535]))}
        refl, siml = compare_dict(d, "test_fhd_many", self.s, self.p, **fhd_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_many_dict(self):
        """Many state changes on OCRA1 gradient outputs, default SPI clock divisor - simultaneous with similar TX changes. Dict version"""
        set_grad_board("ocra1")
        d = {'tx0_i': (np.array([600, 900, 1200, 1500, 1800, 2100, 2400, 2700]),
                          np.array([1, 5, 9, 13, 65535, 65534, 10000, 20000])),
             'tx0_q': (np.array([600, 1200, 1500, 1800, 2100, 2400, 2700]),
                          np.array([2, 10, 14, 2, 65533, 12000, 25000])),
             'tx1_i': (np.array([600, 900, 1500, 1800, 2100, 2400, 2700]),
                          np.array([3, 7, 15, 3, 65532, 14000, 30000])),
             'tx1_q': (np.array([600, 900, 1200, 1500, 1800, 2100, 2400, 2700]),
                          np.array([4, 8, 12, 16, 4, 65531, 16000, 35000])),

             'ocra1_vx': (np.array([600, 900, 1200, 1500, 1800, 2100, 2400, 2700]),
                          np.array([1, 5, 9, 13, 262143, 262142, 100000, 20000])),
             'ocra1_vy': (np.array([600, 1200, 1500, 1800, 2100, 2400, 2700]),
                          np.array([2, 10, 14, 2, 262141, 120000, 25000])),
             'ocra1_vz': (np.array([600, 900, 1500, 1800, 2100, 2400, 2700]),
                          np.array([3, 7, 15, 3, 262140, 140000, 30000])),
             'ocra1_vz2': (np.array([600, 900, 1200, 1500, 1800, 2100, 2400, 2700]),
                          np.array([4, 8, 12, 16, 4, 262139, 160000, 35000])),
             }
        refl, siml = compare_dict(d, "test_oc1_many", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_oc1_two_dict(self):
        """Many state changes on OCRA1 gradient outputs, default SPI clock divisor - attempt to probe dict sorting bug."""
        set_grad_board("ocra1")
        d = {'ocra1_vx': (np.array([600, 900]), np.array([0x10001, 0x20002])),
             'ocra1_vy': (np.array([600]), np.array([0x10001]))}
        refl, siml = compare_dict(d, "test_oc1_two_dict", self.s, self.p, **oc1_config)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_single_expt(self):
        """ Basic state change on a single buffer. Experiment version"""
        set_grad_board("gpa-fhdo")
        d = {'tx0_i': (np.array([1]), np.array([0.5]))}
        expt_args = {'rx_t': 2, 'auto_leds': False}
        refl, siml = compare_expt_dict(d, "test_single_expt", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

        # test for the other grad board
        self.tearDown(); self.setUp()
        set_grad_board("ocra1")
        refl, siml = compare_expt_dict(d, "test_single_expt", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_four_par_expt_iq(self):
        """ State change on four buffers in parallel. Experiment version using complex inputs"""
        d = {'tx0': (np.array([1]), np.array([0.5+0.2j])), 'tx1': (np.array([1]), np.array([-0.3+1j]))}
        expt_args = {'rx_t': 2, 'auto_leds': False}
        set_grad_board("gpa-fhdo")
        refl, siml = compare_expt_dict(d, "test_four_par_expt_iq", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

        self.tearDown(); self.setUp()
        set_grad_board("ocra1")
        refl, siml = compare_expt_dict(d, "test_four_par_expt_iq", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_uneven_sparse_expt_fhd(self):
        """ Miscellaneous pulses on TX and gradients, with various acquisition windows """
        d = {'tx0': (np.array([10,15, 30,35, 100,105]), np.array([1,0, 0.8j,0, 0.7+0.2j,0])),
             'tx1': (np.array([5,20,  50,70,  110,125]), np.array([-1j,0,  -0.5j,0,  0.5+0.3j,0])),

             # gradient events may occur only on a single channel at a
             # time (i.e. no parallel updates) and must be spaced by
             # at least 4us - i.e. after any update occurs, the next
             # update (on any channel) may only occur 4us later
             'grad_vx': (np.array([ 10, 30, 54, 75, 100]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vy': (np.array([ 14, 26, 50, 71,  96]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vz': (np.array([ 6,  22,     79,  88]), np.array([-1, 1,       0.5, 0])),
             'grad_vz2': (np.array([18,     46, 67,  92]), np.array([-1,      1,  0.5, 0])),

             'rx0_en': (np.array([7,12,  30,40,  80,90]), np.array([1,0, 1,0, 1,0])),
             'rx1_en': (np.array([8,14,  33,45,  83,95]), np.array([1,0, 1,0, 1,0])),
             }

        set_grad_board("gpa-fhdo")
        expt_args = {'rx_t': 0.5, 'auto_leds': False}
        refl, siml = compare_expt_dict(d, "test_uneven_sparse_expt_fhd", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_uneven_sparse_expt_oc1(self):
        """ Miscellaneous pulses on TX and gradients, with various acquisition windows """
        d = {'tx0': (np.array([10,15, 30,35, 100,105]), np.array([1,0, 0.8j,0, 0.7+0.2j,0])),
             'tx1': (np.array([5,20,  50,70,  110,125]), np.array([-1j,0,  -0.5j,0,  0.5+0.3j,0])),

             # gradient events must be spaced by at least 4us -
             # i.e. after any update occurs, the next update (on any
             # channel) may only occur 4us later
             'grad_vx': (np.array([10, 30, 43, 50, 77]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vy': (np.array([15, 23, 43, 57, 84]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vz': (np.array([15, 23,     65, 77]), np.array([-1, 1, 0.5, -0.5, 0])),
             'grad_vz2': (np.array([15,     43, 65, 77]), np.array([-1, 1, 0.5, -0.5, 0])),

             'rx0_en': (np.array([7,12,  30,40,  80,90]), np.array([1,0, 1,0, 1,0])),
             'rx1_en': (np.array([8,14,  33,45,  83,95]), np.array([1,0, 1,0, 1,0])),
             }

        set_grad_board("ocra1")
        expt_args = {'rx_t': 0.5, 'auto_leds': False}
        refl, siml = compare_expt_dict(d, "test_uneven_sparse_expt_oc1", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_cic_shift_expt(self):
        """ Test the experiment code to program the open-source (non-Xilinx) variant of the CIC filter """
        # d = {'tx0': ( np.array([10, 15]), np.array([0.5, 0]) )}
        set_grad_board("ocra1")
        expt_args = {'rx_t': 0.5, 'set_cic_shift': True, 'auto_leds': False}
        refl, siml = compare_expt_dict({}, "test_cic_shift_expt", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_init_grad_expt_fhd(self):
        """ Test whether GPA-FHDO gradient events happening at time = 0 cause errors -- they should not if there's a sufficient initial wait"""
        set_grad_board("gpa-fhdo")
        expt_args = {'rx_t': 0.5, 'grad_max_update_rate': 0.1, 'auto_leds': False} # deliberately slow update rate
        d = {'grad_vx': (np.array([0, 5]), np.array([0.5, 0]))}
        refl, siml = compare_expt_dict(d, "test_init_grad_expt_fhd", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_init_grad_expt_oc1(self):
        """ Test whether OCRA1 gradient events happening at time = 0 cause errors -- they should not if there's a sufficient initial wait."""
        set_grad_board("ocra1")
        expt_args = {'rx_t': 0.5, 'grad_max_update_rate': 0.1, 'auto_leds': False} # deliberately slow update rate
        d = {'grad_vx': (np.array([0, 20]), np.array([0.5, 0])) }
        refl, siml = compare_expt_dict(d, "test_init_grad_expt_oc1", self.s, self.p, **expt_args)
        restore_grad_board()
        self.assertEqual(refl, siml)

    def test_auto_leds_expt(self):
        """ Test whether the auto-LED scan works correctly """
        expt_args = {'auto_leds': True}
        d = {'tx0_i': (np.array([0, 100]), np.array([1, 0])) }
        refl, siml = compare_expt_dict(d, "test_auto_leds_expt", self.s, self.p, **expt_args)
        self.assertEqual(refl, siml)

    def test_tx_complex_expt(self):
        """ Test whether the Experiment class removes duplicate i/q entries for complex TX data """
        max_rem_instr = mc.max_removed_instructions
        mc.max_removed_instructions = 1
        expt_args = {'auto_leds': False}
        d = {'tx0': (np.array([0, 10, 15, 20, 30, 50]), np.array([1-1j, 1+1j, 1j, -1+1j, -1, -1-1j]))}
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=mc.MarRemovedInstructionWarning)
            refl, siml = compare_expt_dict(d, "test_tx_complex_expt", self.s, self.p, **expt_args)

        # restore to default
        mc.max_removed_instructions = max_rem_instr
        self.assertEqual(refl, siml)

    def test_lo_change_expt(self):
        """ Test whether the Experiment class can handle changes in LO frequency, followed by being rerun """
        expt_args = {'lo_freq': 1, 'auto_leds': False}
        d = {'tx0': (np.array([0, 1]), np.array([0.5, 0])), 'rx0_en': (np.array([2, 3]), np.array([1, 0]))}

        def change_lo(e):
            e.run() # compile internally
            e.set_lo_freq(2)
            return e.run() # compile internally

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=RuntimeWarning) # catch GPA-FHDO error due to re-initialisation
        refl, siml = compare_expt_dict(d, "test_lo_change_expt", self.s, self.p, run_fn=change_lo, **expt_args)
        self.assertEqual(refl, siml)

if __name__ == "__main__":
    unittest.main()
