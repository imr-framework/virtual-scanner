"""
This script is a unit test script for testing T1 and T2 mapping
Author: Enlin Qian
Date: 03/12/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import T1_mapping as dicom2mapT1
import T2_mapping as dicom2mapT2

'''
TI = [0.021, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
TR = [10, 10, 10, 10, 10, 10, 10]
TE = [0.012, 0.022, 0.042, 0.062, 0.102, 0.152, 0.202]
'''
'''C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements/T1T2Mapping/Data/data_new_128x128/T1'''
'''C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements'
    '/T1T2Mapping/Data/T1_IRSE_data/gray_scale_dicom'''
#
T1_map = dicom2mapT1.main(
    dicom_file_path='C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements/T1T2Mapping/Data/data_new_128x128/T1_original_data',
    TR='10, 10, 10, 10, 10, 10, 10',
    TI='0.021, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2',
    TE='0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012',
    pat_id='9306'
    )


# T2_map = dicom2mapT2.main(
#     dicom_file_path='C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements/T1T2Mapping/Data/data_new_128x128/T2_original_data',
#     TR='10, 10, 10, 10, 10, 10, 10',
#     TE='0.012, 0.022, 0.042, 0.062, 0.102, 0.152, 0.202',
#     pat_id='9306')


