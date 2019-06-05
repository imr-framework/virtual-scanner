"""
This script does T2 mapping of dicom images

Author: Enlin Qian
Date: 04/29/2019
Version 2.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.optimize import curve_fit
import os
import time


def main(dicom_file_path: str, TR: str, TE: str, pat_id: str):
    """
    Return T2 mapping of a series of SE images with variable TE.

    Parameters
    ----------
    dicom_file_path: folder path where all dicom files are
    TE: TI values used in SE experiments
    TR: TR values used in SE experiments, should be constant

    Returns
    -------
    T2_map: T2 map generated based on input images and TE TR values

    """
    TR = np.fromstring(TR, dtype=int, sep=',')
    TE = np.fromstring(TE, dtype=float, sep=',')
    TR = TR/1000
    TE = TE/1000
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dicom_file_path):
        for filename1 in fileList:
            if ".dcm" in filename1.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename1))

    ref_image = pydicom.read_file(lstFilesDCM[0])  # Get ref file
    image_size = (int(ref_image.Rows), int(ref_image.Columns), len(lstFilesDCM))  # Load dimensions
    image_data_final = np.zeros(image_size, dtype=ref_image.pixel_array.dtype)

    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)  # read the file, data type is uint16 (0~65535)
        image_data_final[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    image_data_final = image_data_final.astype(np.float64)  # convert data type

    image_data_final = np.divide(image_data_final, np.amax(image_data_final))
    T2_map = np.zeros([image_size[0], image_size[1]])
    p0 = (0.8002804688888, 0.141886338627215, 0.421761282626275, 0.915735525189067)  # initial guess for parameters

    for n2 in range(image_size[0]):
        for n3 in range(image_size[1]):
            y_data = image_data_final[n2, n3, :]
            if 0 not in y_data:
                popt, pcov = curve_fit(T2_sig_eq, (TE, TR), y_data, p0, bounds=(0, 6))
                T2_map[n2, n3] = popt[2]

    T2_map[T2_map > 2] = 2
    # plt.figure()
    # imshowobj = plt.imshow(T2_map, cmap='hot')
    # imshowobj.set_clim(0, 2)
    # plt.show()

    timestr = time.strftime("%Y%m%d%H%M%S")


    mypath1='./src/coms/coms_ui/static/ana/outputs/'+ pat_id
    mypath2='./src/server/ana/outputs/' + pat_id +'/T2_map'

    if not os.path.isdir(mypath1):
        os.makedirs(mypath1)
    if not os.path.isdir(mypath2):
        os.makedirs(mypath2)

    plt.figure(frameon=False)
    plt.imshow(T2_map, cmap='hot')
    cb = plt.colorbar()
    cb.set_label('Time (s)')
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(mypath1 +'/T2_map' + timestr + '.png', bbox_inches='tight', pad_inches=0)

    filename1 = "T2_map" + timestr + ".png"

    pixel_array = (T2_map/2)*65535
    pixel_array_int = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array_int.tostring()
    ds.save_as(mypath2 +'/T2_map' + timestr +'.dcm')

    return filename1, mypath2


def T2_sig_eq(X, a, b, c, d):
    """
    Generate an exponential function for curve fitting

    Parameters
    ----------
    x: independent variables
    y: independent variables
    a: curve fitting parameters
    b: curve fitting parameters
    c: curve fitting parameters
    d: curve fitting parameters

    Returns
    -------
    exponential function used for T2 curve fitting

    """
    x, y = X
    return a * (1 - np.exp(-y / b)) * np.exp(-x / c) + d

