"""
This script is a unit test script for testing ROI analysis of T1 and T2 maps
Author: Enlin Qian
Date: 04/02/2019
Version 1.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import ROI_analysis as dicomROIanalysis

dicomROIanalysis.main(
    dicom_map_path='C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements/ROI Analysis/Data_Manual/T2_SE_map_data',
    frequency_range_lb = 0.004,
    frequency_range_ub = 6.1,
    connectivity = 4,
    sensitivity = 0.98)




