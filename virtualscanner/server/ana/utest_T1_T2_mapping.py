# Copyright of the Board of Trustees of Columbia University in the City of New York

import sys
import unittest
import virtualscanner.server.ana.T1_mapping as dicom2mapT1
import virtualscanner.server.ana.T2_mapping as dicom2mapT2
import numpy as np
from scipy.misc import imread
from virtualscanner.utils import constants
import imageio

SERVER_T1_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_orig_data'
SERVER_T2_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_orig_data'
SERVER_T1_WINDOWS_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_map_windows'
SERVER_T1_MAC_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_map_mac'
SERVER_T2_WINDOWS_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map_windows'
SERVER_T2_MAC_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map_mac'
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
        map_name, dicom_path = dicom2mapT1.main(dicom_file_path=SERVER_T1_INPUT_PATH , TR=TRstr, TE=TEstr, TI=TIstr, pat_id='9306')
        generated_map = imageio.imread(COMS_MAP_PATH / map_name)
        utest_map = imageio.imread(dicom_map_path / 'utest_T1_map.png')

        np.testing.assert_allclose(generated_map, utest_map)

    def test_T2_mapping(self):
        """
        Unit test T2 mapping from SE experiments with 7 different TE values.
        """
        platform = {'win32': SERVER_T2_WINDOWS_MAP_PATH,
                    'darwin': SERVER_T2_MAC_MAP_PATH,
                    'linux': SERVER_T2_MAC_MAP_PATH}
        dicom_map_path = platform[sys.platform]
        map_name, dicom_path = dicom2mapT2.main(dicom_file_path=SERVER_T2_INPUT_PATH, TR=TRstr, TE=TEstr, pat_id='9306')
        generated_map = imageio.imread(COMS_MAP_PATH / map_name)
        utest_map = imageio.imread(dicom_map_path / 'utest_T2_map.png')

        np.testing.assert_allclose(generated_map, utest_map)


if __name__ == '__main__':
    unittest.main()
