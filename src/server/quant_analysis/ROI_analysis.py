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
from scipy.spatial import distance
import numpy.matlib

def main(dicom_map_path: str, intensity_range_lb: np, intensity_range_ub: np, map_size: np):  # lb: lower bound, ub: upper bound
    """
    Return ROI analysis of a ISMRM/NIST phantom

    Parameters
    ----------
    dicom_map_path: folder path where all dicom maps are
    map_type: specify if it is a T1 or T2 map
    intensity_range: intensity range of desired ROIs, pixels out of this range will be set to 0
    map_size: size of map, for example, 256 means the size of map is 256x256

    Returns
    -------
    centers: centers and radii for all spheres in sphere number order
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

    map_data_final_gray = np.matrix.copy(map_data_final[:, :, 0])  # set up a new variable used for image processing and circle detection
    num_spheres = 14
    pixel_indices = np.logical_or((map_data_final_gray <= intensity_range_lb), (map_data_final_gray >= intensity_range_ub))
    map_data_final_gray[pixel_indices] = 0  # threshold so that pixels outside frequency ranges are 0

    map_data_final_gray = np.divide(map_data_final_gray, np.amax(map_data_final_gray))
    map_data_final_gray = np.multiply(map_data_final_gray, 255)  # conversion to grayscale for open cv operations

    map_data_final_gray = map_data_final_gray.astype(np.uint8)  # open cv requires uint8 grayscale images
    circles = cv2.HoughCircles(map_data_final_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=13,
                               param1=150, param2=8, minRadius=5, maxRadius=7)  # initial guess of spheres

    centers = circles[0, :, :2]
    radii = 0.8*np.matlib.repmat(np.amin(circles[0, :, 2]), num_spheres, 1)

    location_map = np.zeros((image_size[0], image_size[1], num_spheres))
    for n2 in range(num_spheres):
        for n3 in range(image_size[0]):
            for n4 in range(image_size[1]):
                location_map[n3, n4, n2] = np.sqrt(np.square(centers[n2, 1]-n3-1)+np.square(centers[n2, 0]-n4-1))  # create a meshgrid (space for optimization)

    mean_all = np.zeros([num_spheres, 1])
    for n5 in range(num_spheres):
        temp1 = location_map[:, :, n5]
        mean_all[n5] = np.mean(map_data_final[temp1 <= radii[n5, 0]])  # retrieve points inside the circles

    indices = np.argmax(mean_all)
    sphere_1_loc = np.concatenate((centers[indices], radii[indices]))  # find the loc and radius of sphere 1

    scale = map_size / 256
    direction_angle_mat = np.array([[0, 0],
                                   [scale*19.6764, 90-18.1284],
                                   [scale*35.9758, 90-35.9850],
                                   [scale*48.8824, 90-54.2402],
                                   [scale*58.0841, 90-72.4385],
                                   [scale*60.9979, -(90-89.9440)],
                                   [scale*57.7023, -(90-71.7044)],
                                   [scale*49.0630, -(90-54.1265)],
                                   [scale*35.1427, -(90-34.8566)],
                                   [scale*18.4985, -(90-17.2355)],
                                   [scale*44.5752, 90-74.4948],
                                   [scale*22.2815, -(90-55.8365)],
                                   [scale*21.8366, 90-56.2077],
                                   [scale*43.8830, -(90-73.3844)]]) # matrix with distance and angle for each sphere respect to sphere 1

    sphere_all_loc = np.zeros([num_spheres, 3])
    sphere_all_loc[0, :] = sphere_1_loc
    for n6 in range(1, num_spheres):
        sphere_all_loc[n6, 0] = direction_angle_mat[n6, 0]*np.sin(np.pi*direction_angle_mat[n6, 1]/180)+sphere_all_loc[0,0]
        sphere_all_loc[n6, 1] = direction_angle_mat[n6, 0]*np.cos(np.pi*direction_angle_mat[n6, 1]/180)+sphere_all_loc[0,1]
        sphere_all_loc[n6, 2] = sphere_all_loc[0, 2]

    plt.figure()
    plt.imshow(map_data_final[:, :, 0], cmap='hot')
    ax = plt.gca()

    for i in sphere_all_loc[:]:
        # draw the center of the circle
        circle_plot = plt.Circle((i[0], i[1]), i[2], color='r', fill=False)
        ax.add_artist(circle_plot)
    plt.show()

    










