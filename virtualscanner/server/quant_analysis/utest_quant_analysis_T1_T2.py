"""
This script is a unit test script for testing T1 and T2 mapping
Author: Enlin Qian
Date: 03/12/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('Virtual-Scanner') + len('Virtual-Scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)

import virtualscanner.server.quant_analysis.T1_mapping as dicom2mapT1

'''
TI = [0.021, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
TR = [10, 10, 10, 10, 10, 10, 10]
TE = [0.012, 0.022, 0.042, 0.062, 0.102, 0.152, 0.202]
'''

T1_map = dicom2mapT1.main(
    dicom_file_path='C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements'
                    '/T1T2Mapping/Data/T1_IRSE_data/gray_scale_dicom',
    TI=[0.021, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
    TR=[10, 10, 10, 10, 10, 10, 10])

'''
T2_map = dicom2mapT2.main(
    dicom_file_path='C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements'
    '/T1T2Mapping/Data/T2_SE_data/gray_scale_dicom',
    TE=[0.012, 0.022, 0.042, 0.062, 0.102, 0.152, 0.202],
    TR=[10, 10, 10, 10, 10, 10, 10])
'''
