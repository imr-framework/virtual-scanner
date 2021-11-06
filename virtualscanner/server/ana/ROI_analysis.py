# Copyright of the Board of Trustees of Columbia University in the City of New York

import os
import time
from pathlib import Path

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

import cv2

#import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np
import numpy.matlib
import pydicom


from virtualscanner.utils import constants

COMS_ANALYZE_PATH = constants.COMS_UI_PATH


def circle_analysis(circles, map_size):
    """
    Return locations of pixels inside a circle.

    Parameters
    ----------
    circles : numpy.ndarray
        1x3 matrix (center_x, center_y, radius)
    map_size : int
        size of parameter map, for example, 256 means the size of map is 256x256

    Returns
    -------
    sphere_map : numpy.ndarray
        Binary map where pixels inside the circle is true and outside of the circle is false
    """

    X, Y = np.meshgrid(np.arange(map_size), np.arange(map_size))
    distance_map = np.sqrt(np.square(X - circles[0]) + np.square(Y - circles[1]))
    sphere_map = distance_map <= circles[2]
    return sphere_map


def main(dicom_map_path: Path, map_type: str, map_size: str, fov: str,
         pat_id: str):  # fov: should be in mm
    """
    Return ROI analysis of a ISMRM/NIST phantom, all 14 spheres are detected.

    Parameters
    ----------
    dicom_map_path : path
        Path of folder where dicom files reside
    map_type : str
        Type of map (T1 or T2)
    map_size : str
        Size of map, for example, 128 means the size of map is 128x128
    fov : str
        Field of view used in experiments
    pat_id : str
        Primary key in REGISTRATION table

    Returns
    -------
    centers : numpy.ndarray
        Centers and radii for all spheres in sphere number order
    sphere_mean : numpy.ndarray
        Mean values for all spheres in sphere number order
    sphere_std : numpy.ndarray
        Std for all spheres in sphere number order
    """
    map_size = np.ndarray.item(np.fromstring(map_size, dtype=int, sep=','))
    fov = np.ndarray.item(np.fromstring(fov, dtype=int, sep=','))
    num_spheres = 14
    voxel_size = fov / map_size  # unit should be mm/voxel
    sphere_all_loc_template = np.zeros([num_spheres, 2])
    rot_center = np.array([[map_size / 2, map_size / 2]])
    distance_angle_mat = np.array([[0, 0],  # first one is Sphere 1, unit mm, degree
                                   [32.214, 75],
                                   [60.687, 53.13],
                                   [83.562, 35.1],
                                   [98.512, 16.32],
                                   [103.81, -1.426],
                                   [99.2, -19.712],
                                   [85.02, -36.8],
                                   [62.305, -55.13],
                                   [31.451, -72.27],
                                   [36.841, -34.228],
                                   [37.093, 36.379],
                                   [76.805, 14.367],
                                   [76.252, -17.5565]])

    if map_type == 'T1':
        max_values = 5
        true_value = np.array(
            [1.838, 1.398, 0.9983, 0.7258, 0.5091, 0.367, 0.2587, 0.1847, 0.1308, 0.0909, 0.0642, 0.04628, 0.03265, 0.02295])
        sphere_1_ref_mean = true_value(0)
        plane_radius = 170  # mm

    if map_type == 'T2':
        max_values = 2  # The maximum value for map should be fixed, any intensity outside that range is meaningless
        true_value = np.array(
            [0.6458, 0.4236, 0.286, 0.1848, 0.1341, 0.0944, 0.06251, 0.04498, 0.03095, 0.0201, 0.0154, 0.01085, 0.007591, 0.00535])
        sphere_1_ref_mean = true_value(0)
        plane_radius = 195  # mm

    tolerance = 0.1
    intensity_range_lb = sphere_1_ref_mean * (1 - tolerance)
    intensity_range_ub = sphere_1_ref_mean * (1 + tolerance)
    distance_angle_mat[:, 0] = distance_angle_mat[:, 0] / voxel_size # convert to pixel distance

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

    for n2 in range(1,
                    num_spheres): # get all sphere centers in the template (use distance instead of just coordinates because of scaling)
        sphere_all_loc_template[n2, 0] = distance_angle_mat[n2, 0] * np.sin(np.pi * distance_angle_mat[n2, 1] / 180) + \
                                         sphere_all_loc_template[0, 0]
        sphere_all_loc_template[n2, 1] = distance_angle_mat[n2, 0] * np.cos(np.pi * distance_angle_mat[n2, 1] / 180) + \
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

        # circles = cv2.HoughCircles(map_data_final_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=13,
        #                            param1=150, param2=6, minRadius=5, maxRadius=7)  # initial guess of spheres
        edges = canny(map_data_final_gray, sigma=3, low_threshold=10, high_threshold=50)
        hough_radii = np.arange(5, 8, 0.1)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=13, min_ydistance=13,
                                                   total_num_peaks=14)
        circles = np.zeros([1, 14, 3])
        circles[0, :, :] = np.concatenate((cx[:, np.newaxis], cy[:, np.newaxis], radii[:, np.newaxis]), axis=1)
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
    mypath = COMS_ANALYZE_PATH / 'static' / 'ana' / 'outputs' / pat_id

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
        plt.savefig(str(mypath) + '/map_with_ROI' + timestr + '.png', bbox_inches='tight', pad_inches=0)
        # , facecolor = fig.get_facecolor(), edgecolor = 'none'

    filename = 'map_with_ROI' + timestr + '.png'
    return filename
