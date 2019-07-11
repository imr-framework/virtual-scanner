# Copyright of the Board of Trustees of  Columbia University in the City of New York

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.optimize import curve_fit

from virtualscanner.utils import constants

SERVER_ANALYZE_PATH = constants.SERVER_ANALYZE_PATH
COMS_ANALYZE_PATH = constants.COMS_UI_PATH


def main(dicom_file_path: Path, TR: str, TE: str, pat_id: str):
    """
    Curve fitting a series of SE images with respect to variable TE values to generate a T2 map.

    Parameters
    ----------
    dicom_file_path : path
        Path of folder where dicom files reside
    TR : str
        TR value used in SE experiments (unit in milliseconds, should be constant)
    TE : str
        TE value used in SE experiments (unit in milliseconds)
    pat_id : str
        primary key in REGISTRATION table

    Returns
    -------
    png_map_name : str
        File name of T2 map in png format
    dicom_map_path : str
        Path of T2 map in dicom format
    """
    TR = np.fromstring(TR, dtype=int, sep=',')
    TE = np.fromstring(TE, dtype=float, sep=',')
    TR = TR / 1000
    TE = TE / 1000
    fov = 210
    map_size = 128
    voxel_size = fov / map_size  # unit should be mm/voxel
    phantom_radius = 101.72  # unit in mm
    centers = np.array([[map_size / 2, map_size / 2]])

    lstFilesDCM = sorted(list(dicom_file_path.glob('*.dcm')))  # create an empty list
    ref_image = pydicom.read_file(str(lstFilesDCM[0]))  # Get ref file
    image_size = (int(ref_image.Rows), int(ref_image.Columns), len(lstFilesDCM))  # Load dimensions
    image_data_final = np.zeros(image_size, dtype=ref_image.pixel_array.dtype)

    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(str(filenameDCM))  # read the file, data type is uint16 (0~65535)
        image_data_final[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    image_data_final = image_data_final.astype(np.float64)  # convert data type

    image_data_final = np.divide(image_data_final, np.amax(image_data_final))
    T2_map = np.zeros([image_size[0], image_size[1]])
    # p0 = (0.8002804688888, 0.141886338627215, 0.421761282626275, 0.915735525189067)  # initial guess for parameters
    p0 = (0.8002804688888, 0.141886338627215, 0.421761282626275)  # initial guess for parameters
    for n2 in range(image_size[0]):
        for n3 in range(image_size[1]):
            dist_to_center = np.sqrt((n2-centers[0, 0]) ** 2+(n3-centers[0, 1]) ** 2) * voxel_size
            y_data = image_data_final[n2, n3, :]
            if dist_to_center < phantom_radius:
                popt, pcov = curve_fit(T2_sig_eq, (TE, TR), y_data, p0, bounds=(0, 2))
                T2_map[n2, n3] = popt[2]

    T2_map[T2_map > 2] = 2
    # plt.figure()
    # imshowobj = plt.imshow(T2_map, cmap='hot')
    # imshowobj.set_clim(0, 2)
    # plt.show()

    timestr = time.strftime("%Y%m%d%H%M%S")
    png_map_path = COMS_ANALYZE_PATH / 'static' / 'ana' / 'outputs' / pat_id
    dicom_map_path = SERVER_ANALYZE_PATH / 'outputs' / pat_id / 'T2_map'

    if not os.path.isdir(png_map_path):
        os.makedirs(png_map_path)
    if not os.path.isdir(dicom_map_path):
        os.makedirs(dicom_map_path)

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
    plt.savefig(str(png_map_path) + '/T2_map' + timestr + '.png', bbox_inches='tight', pad_inches=0)

    png_map_name = "T2_map" + timestr + ".png"

    pixel_array = (T2_map / 2) * 65535
    pixel_array_int = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array_int.tostring()
    ds.save_as(str(dicom_map_path) + '/T2_map' + timestr + '.dcm')

    return png_map_name, dicom_map_path


def T2_sig_eq(X, a, b, c):
    """
    Generate an exponential function for curve fitting.

    Parameters
    ----------
    X : float
        Independent variable
    a : float
        Curve fitting parameters
    b : float
        Curve fitting parameters
    c : float
        Curve fitting parameters

    Returns
    -------
    float
        Exponential function used for T2 curve fitting
    """

    x, y = X
    return a * (1 - np.exp(-y / b)) * np.exp(-x / c)
