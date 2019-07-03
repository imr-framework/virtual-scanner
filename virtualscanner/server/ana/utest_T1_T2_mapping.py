# Copyright of the Board of Trustees of Columbia University in the City of New York

import unittest
import virtualscanner.server.ana.T1_mapping as dicom2mapT1
import virtualscanner.server.ana.T2_mapping as dicom2mapT2
import numpy as np
from scipy.misc import imread
from virtualscanner.utils import constants

SERVER_T1_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_orig_data'
SERVER_T2_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_orig_data'
SERVER_T1_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_map'
SERVER_T2_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map'
COMS_MAP_PATH = constants.COMS_UI_PATH / 'static' / 'ana' / 'outputs' / '9306'


TIstr = '21, 100, 200, 400, 800, 1600, 3200'
TRstr = '10000, 10000, 10000, 10000, 10000, 10000, 10000'
TEstr = '12, 22, 42, 62, 102, 152, 202'


class MyTestCase(unittest.TestCase):
    def test_T1_mapping(self):
        """
        Unit test T1 mapping from IRSE experiments with 7 different TI values.
        """
        map_name, dicom_path = dicom2mapT1.main(dicom_file_path=SERVER_T1_INPUT_PATH , TR=TRstr, TE=TEstr, TI=TIstr, pat_id='9306')
        generated_map = imread(COMS_MAP_PATH / map_name)
        utest_map = imread(SERVER_T1_MAP_PATH / 'utest_T1_map.png')

        np.testing.assert_allclose(generated_map, utest_map)

    def test_T2_mapping(self):
        """
        Unit test T2 mapping from SE experiments with 7 different TE values.
        """
        map_name, dicom_path = dicom2mapT2.main(dicom_file_path=SERVER_T2_INPUT_PATH, TR=TRstr, TE=TEstr, pat_id='9306')
        generated_map = imread(COMS_MAP_PATH / map_name)
        utest_map = imread(SERVER_T2_MAP_PATH / 'utest_T2_map.png')

        np.testing.assert_allclose(generated_map, utest_map)


if __name__ == '__main__':
    unittest.main()
