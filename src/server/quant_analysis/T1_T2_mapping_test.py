import T2_mapping as dicom2map

'''
TI = [0.021, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
TR = [10, 10, 10, 10, 10, 10, 10]
TE = [0.012, 0.022, 0.042, 0.062, 0.102, 0.152, 0.202]
'''

T2_map = dicom2map.main(
    dicom_file_path='C:/Users/qiane/Desktop/Columbia/Research Materials/Programs/Phantom Standard Measurements'
    '/T1T2Mapping/Data/T2_SE_data',
    TE=[0.012, 0.022, 0.042, 0.062, 0.102, 0.152, 0.202],
    TR=[10, 10, 10, 10, 10, 10, 10])
