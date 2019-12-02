# Copyright of the Board of Trustees of Columbia University in the City of New York

import sys
import unittest
import virtualscanner.server.ana.T1_mapping as dicom2mapT1
import virtualscanner.server.ana.T2_mapping as dicom2mapT2
import numpy as np
from virtualscanner.utils import constants
import imageio

SERVER_T1_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_orig_data'
SERVER_T2_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_orig_data'
SERVER_T1_WINDOWS_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_map_windows'
SERVER_T1_MAC_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_map_mac'
SERVER_T2_WINDOWS_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map_windows'
SERVER_T2_MAC_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map_mac'
SERVER_T1_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'outputs' / '9306' / 'T1_map'
SERVER_T2_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'outputs' / '9306' / 'T2_map'
COMS_MAP_PATH = constants.COMS_UI_PATH / 'static' / 'ana' / 'outputs' / '9306'


TIstr = '21, 100, 200, 400, 800, 1600, 3200'
TRstr = '10000, 10000, 10000, 10000, 10000, 10000, 10000'
TEstr = '12, 22, 42, 62, 102, 152, 202'


class MyTestCase(unittest.TestCase):
    def test_T1_mapping(self):
        """
        Unit test T1 mapping from IRSE experiments with 7 different TI values.
        """
        platform = {'win32': SERVER_T1_WINDOWS_MAP_PATH,
                    'darwin': SERVER_T1_MAC_MAP_PATH,
                    'linux': SERVER_T1_MAC_MAP_PATH}
        dicom_map_path = platform[sys.platform]
        map_name, dicom_path, np_map_name = dicom2mapT1.main(dicom_file_path=SERVER_T1_INPUT_PATH , TR=TRstr, TE=TEstr, TI=TIstr, pat_id='9306')
        generated_map = np.load(SERVER_T1_MAP_PATH / np_map_name)
        utest_map = np.load(dicom_map_path / 'utest_T1_map.npy')

        # python_version = sys.version_info
        # if python_version.major == 3 and python_version.minor == 6 and python_version.micro == 9:
        #     rtol = 30
        #     atol = 10
        # else:
        #     rtol = 1e-7
        #     atol = 0
        # np.testing.assert_allclose(generated_map, utest_map, rtol, atol)
        np.testing.assert_allclose(generated_map, utest_map)

    # def test_T2_mapping(self):
    #     """
    #     Unit test T2 mapping from SE experiments with 7 different TE values.
    #     """
    #     platform = {'win32': SERVER_T2_WINDOWS_MAP_PATH,
    #                 'darwin': SERVER_T2_MAC_MAP_PATH,
    #                 'linux': SERVER_T2_MAC_MAP_PATH}
    #     dicom_map_path = platform[sys.platform]
    #     map_name, dicom_path, np_map_name = dicom2mapT2.main(dicom_file_path=SERVER_T2_INPUT_PATH, TR=TRstr, TE=TEstr, pat_id='9306')
    #     generated_map = np.load(SERVER_T2_MAP_PATH / np_map_name)
    #     utest_map = np.load(dicom_map_path / 'utest_T2_map.npy')
    #
    #     python_version = sys.version_info
    #     if python_version.major == 3 and python_version.minor == 6 and python_version.micro == 9:
    #         rtol = 30
    #         atol = 10
    #     else:
    #         rtol = 1e-7
    #         atol = 0
    #     np.testing.assert_allclose(generated_map, utest_map, rtol, atol)

if __name__ == '__main__':
    unittest.main()
