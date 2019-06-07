"""
This script does ROI analysis of dicom maps of ISMRM-NIST phantom

Author: Enlin Qian
Date: 04/08/2019
Version 1.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os
import cv2
import numpy.matlib
import time


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


def main(dicom_map_path: str, map_type: str, map_size: str, fov: str, pat_id: str):  # lb: lower bound, ub: upper bound, fov: should be in mm
    """
    Return ROI analysis of a ISMRM/NIST phantom

    Parameters
    ----------
    dicom_map_path: folder path where all dicom maps are
    map_type: specify if it is a T1 or T2 map (used to restore original intensity)
    intensity_range: intensity range of desired ROIs, pixels out of this range will be set to 0
    map_size: size of map, for example, 128 means the size of map is 128x128

    Returns
    -------
    centers: centers and radii for all spheres in sphere number order
    sphere_mean: mean values for all spheres in sphere number order
    sphere_std: std for all spheres in sphere number order
    """
    map_size = np.ndarray.item(np.fromstring(map_size, dtype=int, sep=','))
    fov = np.ndarray.item(np.fromstring(fov, dtype=int, sep=','))
    num_spheres = 14
    voxel_size = fov/map_size  # unit should be mm/voxel
    sphere_all_loc_template = np.zeros([num_spheres, 2])
    if map_type == 'T1':
        max_values = 5
        sphere_1_ref_mean = 1.989
        direction_angle_mat = np.array([[0, 0],  # first one is Sphere 1, unit mm, degree
                                        [voxel_size * 23.9443, 70.0850],
                                        [voxel_size * 44.7587, 53.3538],
                                        [voxel_size * 61.8170, 35.3270],
                                        [voxel_size * 72.2544, 16.9426],
                                        [voxel_size * 75.4064, -0.6324],
                                        [voxel_size * 71.9021, -18.7705],
                                        [voxel_size * 60.8042, -36.9326],
                                        [voxel_size * 44.5492, -56.4287],
                                        [voxel_size * 23.2302, -72.7011],
                                        [voxel_size * 26.6487, 33.6653],
                                        [voxel_size * 27.3647, -34.0284],
                                        [voxel_size * 55.5667, 14.9729],
                                        [voxel_size * 55.0918, -16.4091]])
        sphere_all_loc_template[0, 0:2] = np.squeeze(np.array([[64.8397, 26.1065]]))
        radius_scale = 0.97
        intensity_range_lb = 0
        intensity_range_ub = 5
        golden_standard = np.array([1.989, 1.454, 0.9841, 0.706, 0.4967, 0.3515, 0.24713, 0.1753, 0.1259, 0.089, 0.0627, 0.04453, 0.03084, 0.021719])
    if map_type == 'T2':
        max_values = 2  # The maximum value for map should be fixed, any intensity outside that range is meaningless
        sphere_1_ref_mean = 0.5813
        direction_angle_mat = np.array([[0, 0],  # first one is Sphere 1
                                        [voxel_size * 20.0543, 71.4899],
                                        [voxel_size * 36.5569, 54.0302],
                                        [voxel_size * 49.9120, 35.7043],
                                        [voxel_size * 58.6171, 17.5603],
                                        [voxel_size * 61.5867, -0.1549],
                                        [voxel_size * 58.0576, -18.3830],
                                        [voxel_size * 49.4963, -35.8010],
                                        [voxel_size * 35.5692, -54.2831],
                                        [voxel_size * 18.2029, -70.3638],
                                        [voxel_size * 22.2754, 33.1854],
                                        [voxel_size * 21.9875, -33.0284],
                                        [voxel_size * 44.7193, 15.213],
                                        [voxel_size * 44.2191, -16.2861]])  # matrix with distance and angle for each sphere respect to sphere 1
        sphere_all_loc_template[0, 0:2] = np.squeeze(np.array([[64.1242, 33.5806]]))
        radius_scale = 0.95
        intensity_range_lb = 0
        intensity_range_ub = 2
        golden_standard = np.array([0.5813, 0.4035, 0.2781, 0.19094, 0.13327, 0.09689, 0.06407, 0.04642, 0.03197, 0.02256, 0.015813, 0.011237, 0.007911, 0.005592])

    direction_angle_mat[:, 0] = direction_angle_mat[:, 0]/voxel_size
    direction_angle_mat[:, 1] = direction_angle_mat[:, 1]
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

    map_data_final = map_data_final.astype(np.float64)  # convert data type
    for n1 in range(image_size[2]):
        map_data_final[:, :, n1] = np.divide(map_data_final[:, :, n1], np.amax(map_data_final[:, :, n1]))
        map_data_final[:, :, n1] = np.multiply(map_data_final[:, :, n1],
                                               max_values)  # conversion to original intensity (this is desired)

    rot_center = np.array([[map_size / 2, map_size / 2]])

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
    rot_angle_deg = np.zeros([image_size[2], 1])
    sphere_all_mean_std = np.zeros([num_spheres, 2, image_size[2]])
    map_data_final_with_ROI = np.zeros([map_size, map_size, image_size[2]])
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
                                   param1=150, param2=6, minRadius=5, maxRadius=7)  # initial guess of spheres
        # # visualize the original guess
        # plt.figure()
        # plt.imshow(map_data_final[:, :, 0], cmap='hot')
        # ax = plt.gca()
        #
        # for n6 in range(circles.shape[1]):
        #     # draw the center of the circle
        #     circle_plot = plt.Circle((circles[0, n6, 0], circles[0, n6, 1]),
        #                              circles[0, n6, 2], color='b', fill=False)
        #     ax.add_artist(circle_plot)
        # plt.show()

        circles_all = circles[0, :, :]
        circles_all[:, 2] = np.squeeze(radius_scale * np.matlib.repmat(np.amin(circles[0, :, 2]), circles.shape[1],
                                                              1))  # use smallest radius for calculating means
        circles_mean = np.zeros([circles.shape[1], 1])

        for n4 in range(circles.shape[
                            1]):  # loop through all the circles (may not be same number as num_spheres due to false positives)
            circle_loc = circle_analysis(circles_all[n4, :], map_size)  # get binary map
            map_data_temple = map_data_final[np.nonzero(circle_loc)]
            map_data_temple = map_data_temple[:, n3]
            circles_mean[n4, 0] = np.mean(map_data_temple)

        sphere_1_loc_map[0, :, n3] = circles_all[
            np.argmin(np.abs(circles_mean - sphere_1_ref_mean))]  # find the loc and radius of sphere 1 for all maps
        angle_map_vec = np.array(
            [[sphere_1_loc_map[0, 0, n3] - rot_center[0, 0], sphere_1_loc_map[0, 1, n3] - rot_center[0, 1]]])
        rot_angle = np.arccos(np.divide(np.dot(np.squeeze(angle_template_vec), np.squeeze(angle_map_vec)),
                                        np.linalg.norm(angle_template_vec) * np.linalg.norm(
                                            angle_map_vec)))  # calculate rot angle
        rot_angle_deg[n3, 0] = rot_angle * 180 / np.pi
        rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
        sphere_all_loc_map[:, 0:2, n3] = np.squeeze(np.matmul(sphere_all_loc_template - rot_center,
                                                              rot_mat)) + rot_center  # rotation matrix is respect to origin
        sphere_all_loc_map[:, 2, n3] = np.squeeze(np.matlib.repmat(circles_all[0, 2], num_spheres, 1))

    timestr = time.strftime("%Y%m%d%H%M%S")
    mypath = './src/coms/coms_ui/static/ana/outputs/' + pat_id

    if not os.path.isdir(mypath):
            os.makedirs(mypath)

    # visualize final guess
    for n5 in range(image_size[2]):
        fig = plt.figure(frameon=False)
        im = plt.imshow(map_data_final[:, :, n5], cmap='hot')
        ax = plt.gca()

        for n6 in range(num_spheres):
            # draw the center of the circle
            circle_plot = plt.Circle((sphere_all_loc_map[n6, 0, n5], sphere_all_loc_map[n6, 1, n5]),
                                     sphere_all_loc_map[n6, 2, n5], linewidth=2.5, color='lightgreen', fill=False)
            ax.add_artist(circle_plot)

        cb = plt.colorbar()
        cb.set_label('Time (s)')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(mypath + '/map_with_ROI' + timestr + '.png', bbox_inches='tight', pad_inches = 0)
        # , facecolor = fig.get_facecolor(), edgecolor = 'none'

    filename = 'map_with_ROI' + timestr + '.png'
    return filename
