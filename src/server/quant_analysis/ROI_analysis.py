"""
This script does ROI analysis of dicom maps

Author: Enlin Qian
Date: 04/01/2019
Version 1.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os
import cv2

def main(dicom_map_path: str, frequency_range_lb: np, frequency_range_ub: np, connectivity: np, sensitivity: np):  # lb: lower bound, ub: upper bound
    """
    Return ROI analysis of a ISMRM/NIST phantom

    Parameters
    ----------
    dicom_map_path: folder path where all dicom maps are
    map_type: specify if it is a T1 or T2 map
    frequency_range: frequency range of desired ROIs, pixels out of this range will be set to 0
    connectivity: object connectivity
    sensitivity: circle detection sensitivity

    Returns
    -------
    centers: centers for all spheres in sphere number order
    radii: radius for all spheres in sphere number order
    sphere_mean: mean values for all spheres in sphere number order
    sphere_std: std for all spheres in sphere number order
    """
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dicom_map_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))

    ref_image = pydicom.read_file(lstFilesDCM[0])  # Get ref file
    image_size = (int(ref_image.Rows), int(ref_image.Columns), len(lstFilesDCM))  # Load dimensions
    map_data_final = np.zeros(image_size, dtype=ref_image.pixel_array.dtype)

    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)  # read the file, data type is uint16 (0~65535)
        map_data_final[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    max_values = [5.1892, 0]  # 6 for T1, 5.1892 for T2 map
    map_data_final = map_data_final.astype(np.float64)  # convert data type
    for n1 in range(image_size[2]):
        map_data_final[:, :, n1] = np.divide(map_data_final[:, :, n1], np.amax(map_data_final[:, :, n1]))
        map_data_final[:, :, n1] = np.multiply(map_data_final[:, :, n1], max_values[n1])  # conversion to original intensity (this is desired)

    map_data_final_gray = map_data_final  # set up a new variable used for image processing and circle detection
    num_spheres = 14
    pixel_indices = np.logical_or((map_data_final_gray <= frequency_range_lb), (map_data_final_gray >= frequency_range_ub))
    map_data_final_gray[pixel_indices] = 0  # threshold so that pixels outside frequency ranges are 0

    for n2 in range(image_size[2]):
        map_data_final_gray[:, :, n2] = np.divide(map_data_final[:, :, n2], np.amax(map_data_final[:, :, n2]))
        map_data_final_gray[:, :, n2] = np.multiply(map_data_final_gray[:, :, n2], 255)  # conversion to grayscale for open cv operations
    map_data_final_gray = map_data_final_gray.astype(np.uint8)  # open cv requires uint8 grayscale images
    circles = cv2.HoughCircles(map_data_final_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=12,
                               param1=150, param2=8, minRadius=5, maxRadius=7)

    plt.figure()
    plt.imshow(map_data_final[:, :, 0], cmap='hot')
    ax = plt.gca()
    for i in circles[0, :]:
        # draw the center of the circle
        circle_plot = plt.Circle((i[0], i[1]), i[2], color='r', fill=False)
        ax.add_artist(circle_plot)
    plt.show()  # plot circles to see if they match original image

