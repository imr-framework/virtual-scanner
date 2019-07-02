# Copyright of the Board of Trustees of  Columbia University in the City of New York

import virtualscanner.server.ana.ROI_analysis as dicomROIanalysis
from virtualscanner.utils import constants

SERVER_T2_MAP_INPUT_PATH = constants.SERVER_ANALYZE_PATH / 'inputs' / 'T2_map'

if __name__ == '__main__':
    dicomROIanalysis.main(
        dicom_map_path=SERVER_T2_MAP_INPUT_PATH,
        map_type='T1',
        map_size='128',
        fov='210',
        pat_id='9306')
