"""
This script is a unit test script for testing ROI analysis of T1 and T2 maps
Author: Enlin Qian
Date: 04/02/2019
Version 1.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import virtualscanner.server.ana.ROI_analysis as dicomROIanalysis
from virtualscanner.utils import constants

SERVER_T2_MAP_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map'

dicomROIanalysis.main(
    dicom_map_path=SERVER_T2_MAP_INPUT_PATH,
    map_type='T1',
    map_size='128',
    fov='210',
    pat_id='9306')
