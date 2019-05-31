"""
This script does ROI analysis of dicom maps

Author: Enlin Qian
Date: 04/08/2019
Version 1.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pydicom


def circle_analysis(circles, map_size):
    """
    Return locations of pixels inside a circle

    Parameters
    ----------
    circles: 1x3 matrix (center_x, center_y, radius)
    map_size: size of map, for example, 256 means the size of map is 256x256

    Returns
    -------
    sphere_map: binary map where pixels inside the circle is true and outside of the circle is false
    """
    X, Y = np.meshgrid(np.arange(map_size), np.arange(map_size))
    distance_map = np.sqrt(np.square(X - circles[0]) + np.square(Y - circles[1]))
    sphere_map = distance_map <= circles[2]
    return sphere_map


def main(dicom_map_path: str, intensity_range_lb: np, intensity_range_ub: np,
         map_size: np):  # lb: lower bound, ub: upper bound
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
        map_data_final[:, :, n1] = np.multiply(map_data_final[:, :, n1],
                                               max_values[n1])  # conversion to original intensity (this is desired)

    num_spheres = 14
    scale = map_size / 256
    rot_center = np.array([[128 / 2, map_size / 2]])  # This need to be changed after new experiments are done
    direction_angle_mat = np.array([[0, 0],  # first one is Sphere 1
                                    [scale * 19.6764, 90 - 18.1284],
                                    [scale * 35.9758, 90 - 35.9850],
                                    [scale * 48.8824, 90 - 54.2402],
                                    [scale * 58.0841, 90 - 72.4385],
                                    [scale * 60.9979, -(90 - 89.9440)],
                                    [scale * 57.7023, -(90 - 71.7044)],
                                    [scale * 49.0630, -(90 - 54.1265)],
                                    [scale * 35.1427, -(90 - 34.8566)],
                                    [scale * 18.4985, -(90 - 17.2355)],
                                    [scale * 44.5752, 90 - 74.4948],
                                    [scale * 22.2815, -(90 - 55.8365)],
                                    [scale * 21.8366, 90 - 56.2077],
                                    [scale * 43.8830, -(
                                                90 - 73.3844)]])  # matrix with distance and angle for each sphere respect to sphere 1
    sphere_all_loc_template = np.zeros([num_spheres, 2])
    sphere_all_loc_template[0, 0:2] = np.squeeze(np.array([[62.9708, 98.2607]]))

    for n2 in range(1,
                    num_spheres):  # get all sphere centers in the template (use distance instead of just coordinates because of scaling)
        sphere_all_loc_template[n2, 0] = direction_angle_mat[n2, 0] * np.sin(np.pi * direction_angle_mat[n2, 1] / 180) + \
                                         sphere_all_loc_template[0, 0]
        sphere_all_loc_template[n2, 1] = direction_angle_mat[n2, 0] * np.cos(np.pi * direction_angle_mat[n2, 1] / 180) + \
                                         sphere_all_loc_template[0, 1]
    angle_template_vec = np.array(
        [[sphere_all_loc_template[0, 0] - rot_center[0, 0], sphere_all_loc_template[0, 1] - rot_center[0, 1]]])

    sphere_1_loc_map = np.zeros([1, 3, image_size[2]])
    sphere_all_loc_map = np.zeros([num_spheres, 3, image_size[2]])
    for n3 in range(image_size[2]):  # process all maps in the folder
        map_data_final_gray = np.matrix.copy(
            map_data_final[:, :, n3])  # set up a new variable used for image processing and circle detection
        pixel_indices = np.logical_or((map_data_final_gray <= intensity_range_lb),
                                      (map_data_final_gray >= intensity_range_ub))
        map_data_final_gray[pixel_indices] = 0  # threshold so that pixels outside frequency ranges are 0
        map_data_final_gray = np.divide(map_data_final_gray, np.amax(map_data_final_gray))
        map_data_final_gray = np.multiply(map_data_final_gray, 255)  # conversion to grayscale for open cv operations
        map_data_final_gray = map_data_final_gray.astype(np.uint8)  # open cv requires uint8 grayscale images
        circles = cv2.HoughCircles(map_data_final_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=13,
                                   param1=150, param2=8, minRadius=5, maxRadius=7)  # initial guess of spheres
        circles_all = circles[0, :, :]
        circles_all[:, 2] = np.squeeze(0.8 * np.matlib.repmat(np.amin(circles[0, :, 2]), circles.shape[1],
                                                              1))  # use smallest radius for calculating means
        for n4 in range(circles.shape[
                            1]):  # loop through all the circles (may not be same number as num_spheres due to false positives)
            circle_loc = circle_analysis(circles_all[n4, :], map_size)  # get binary map
            circle_mean = np.mean(map_data_final[np.where(circle_loc)[0]])
            sphere_1_loc_map[0, :, n3] = circles_all[
                np.argmax(circle_mean)]  # find the loc and radius of sphere 1 for all maps

        angle_map_vec = np.array(
            [[sphere_1_loc_map[0, 0, n3] - rot_center[0, 0], sphere_1_loc_map[0, 1, n3] - rot_center[0, 1]]])
        rot_angle = np.arccos(np.divide(np.dot(np.squeeze(angle_template_vec), np.squeeze(angle_map_vec)),
                                        np.linalg.norm(angle_template_vec) * np.linalg.norm(
                                            angle_map_vec)))  # calcualte rot angle
        rot_angle = rot_angle * np.pi / 180
        rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
        sphere_all_loc_map[:, 0:2, n3] = np.squeeze(np.matmul(sphere_all_loc_template, rot_mat))
        sphere_all_loc_map[:, 2, n3] = circles_all[:, 2]

    plt.figure()
    plt.imshow(map_data_final[:, :, 0], cmap='hot')
    ax = plt.gca()

    for n5 in range(num_spheres):
        # draw the center of the circle
        circle_plot = plt.Circle((sphere_all_loc_map[n5, 0, 0], sphere_all_loc_map[n5, 1, 0]),
                                 sphere_all_loc_map[n5, 2, 0], color='r', fill=False)
        ax.add_artist(circle_plot)
    plt.show()
