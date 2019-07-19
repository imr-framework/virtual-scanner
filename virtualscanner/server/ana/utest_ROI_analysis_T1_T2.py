import unittest
import virtualscanner.server.ana.ROI_analysis as dicomROIanalysis
import numpy as np
from scipy.misc import imread
from virtualscanner.utils import constants

SERVER_T1_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T1_map'
SERVER_T2_MAP_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map'
COMS_MAP_PATH = constants.COMS_UI_PATH / 'static' / 'ana' / 'outputs' / '9306'

class MyTestCase(unittest.TestCase):
    def test_T1_ROI_analysis(self):
        """
        Unit test ROI analysis of T1 map.
        """
        roi_filename = dicomROIanalysis.main(dicom_map_path=SERVER_T1_MAP_PATH, map_type='T1', map_size='128',
                                                fov='170', pat_id='9306')
        generated_ROI = imread(COMS_MAP_PATH / roi_filename)
        utest_ROI = imread(SERVER_T1_MAP_PATH / 'utest_T1_map_with_ROI.png')

        np.testing.assert_allclose(generated_ROI, utest_ROI)

    def test_T2_ROI_analysis(self):
        """
        Unit test ROI analysis of T2 map.
        """
        roi_filename = dicomROIanalysis.main(dicom_map_path=SERVER_T2_MAP_PATH, map_type='T2', map_size='128',
                                                fov='210', pat_id='9306')
        generated_ROI = imread(COMS_MAP_PATH / roi_filename)
        utest_ROI = imread(SERVER_T2_MAP_PATH / 'utest_T2_map_with_ROI.png')

        np.testing.assert_allclose(generated_ROI, utest_ROI)


if __name__ == '__main__':
    unittest.main()
