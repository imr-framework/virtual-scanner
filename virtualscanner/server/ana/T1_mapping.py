# Copyright of the Board of Trustees of Columbia University in the City of New York

import os
import time
from pathlib import Path

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.optimize import curve_fit

from virtualscanner.utils import constants

SERVER_ANALYZE_PATH = constants.SERVER_ANALYZE_PATH
COMS_ANALYZE_PATH = constants.COMS_UI_PATH


def main(dicom_file_path: Path, TR: str, TE: str, TI: str, pat_id: str):  # TI should be in second
    """
    Curve fitting a series of IRSE images with respect to variable TI values to generate a T1 map.

    Parameters
    ----------
    dicom_file_path : path
        Path of folder where dicom files reside
    TR : str
        TR value used in IRSE experiments (unit in milliseconds, should be constant)
    TI : str
        TI values used in IRSE experiments (unit in milliseconds)
    TE : str
        TE value used in IRSE experiments (unit in milliseconds, should be constant)
    pat_id : str
        Primary key in REGISTRATION table

    Returns
    -------
    png_map_name : str
        file name of T1_map in png format
    dicom_map_path : str
        path of T1_map in dicom format
    """
    TR = np.fromstring(TR, dtype=int, sep=',')
    TE = np.fromstring(TE, dtype=float, sep=',')
    TI = np.fromstring(TI, dtype=float, sep=',')
    TR = TR / 1000
    TE = TE / 1000
    TI = TI / 1000

    lstFilesDCM = sorted(list(dicom_file_path.glob('*.dcm')))
    ref_image = pydicom.read_file(str(lstFilesDCM[0]))  # Get ref file
    image_size = (int(ref_image.Rows), int(ref_image.Columns), len(
        lstFilesDCM))  # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    image_data_final = np.zeros(image_size, dtype=ref_image.pixel_array.dtype)

    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(str(filenameDCM))  # read the file
        image_data_final[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  # store the raw image data (uint16)
    image_data_final = image_data_final.astype(np.float64)  # convert data type

    # image_data_final = np.divide(image_data_final, np.amax(image_data_final))
    image_data_final = image_data_final/1000
    T1_map = np.zeros([image_size[0], image_size[1]])

    for n2 in range(image_size[0]):
        for n3 in range(image_size[1]):
            y_data = image_data_final[n2, n3, :]
            n4 = 0
            min_loc = np.argmin(y_data)
            while n4 <= min_loc:
                y_data[n4] = -y_data[n4]
                n4 = n4 + 1
            popt, pcov = curve_fit(T1_sig_eq, (TI, TR), y_data,
                                   p0=(0.991966876997438, 0.526367507137759, 0.961046087302673), bounds=([0, 0, -1], [10, 6, 6]))
            T1_map[n2, n3] = popt[1]

    T1_map[T1_map > 5] = 5

    # plt.figure()
    # imshowobj = plt.imshow(T1_map, cmap='hot')
    # imshowobj.set_clim(0, 5)
    # plt.show()

    timestr = time.strftime("%Y%m%d%H%M%S")
    png_map_path = COMS_ANALYZE_PATH / 'static' / 'ana' / 'outputs' / pat_id
    dicom_map_path = SERVER_ANALYZE_PATH / 'outputs' / pat_id / 'T1_map'
    np_map_path = SERVER_ANALYZE_PATH / 'outputs' / pat_id / 'T1_map'

    if not os.path.isdir(png_map_path):
        os.makedirs(png_map_path)
    if not os.path.isdir(dicom_map_path):
        os.makedirs(dicom_map_path)
    if not os.path.isdir(np_map_path):
        os.makedirs(np_map_path)

    np_map_name = 'T1_map' + timestr + '.npy'
    np.save(str(np_map_path) + '/T1_map' + timestr + '.npy', T1_map)

    plt.figure(frameon=False)
    plt.imshow(T1_map, cmap='hot')
    cb = plt.colorbar()
    cb.set_label('Time (s)')
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(str(png_map_path) + '/T1_map' + timestr + '.png', bbox_inches='tight', pad_inches=0)

    # plt.imsave(png_map_path +'/T1_map' + timestr + '.png', T1_map, vmin = 0, vmax = 5, cmap='hot')
    png_map_name = "T1_map" + timestr + ".png"

    pixel_array = (T1_map / 5) * 65535
    pixel_array_int = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array_int.tostring()
    ds.save_as(str(dicom_map_path) + '/T1_map' + timestr + '.dcm')

    return png_map_name, dicom_map_path, np_map_name


def T1_sig_eq(X, a, b, c):
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
        Exponential function used for T1 curve fitting
    """
    x, y = X
    return a * (1 - 2 * np.exp(-x / b) + np.exp(-y / b)) + c
