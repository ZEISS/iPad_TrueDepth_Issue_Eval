# Copyright (C) 2022, Carl Zeiss AG
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of The Regents or University of California nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Please contact the author of this library if you have any questions.
# Author: Steffen Urban (steffen.urban@zeiss.com)

import os
import numpy as np
from ast import literal_eval
import cv2


def check_if_pose_is_close(all_poses, position_to_test, distance):
    """
    Parameters
    ----------
    all_poses : list
        a list of arrays containing camera positions
    position_to_test : ndarray
        a position to test
    distance : float
        a minimum distance to decide if the position_to_test is close to any position
        in all_poses
    Returns
    -------
    bool
        True or False if position_to_test is within close distance to all_poses
    """
    for i in range(len(all_poses)):
        dist = cv2.norm(all_poses[i] - position_to_test)
        if dist < distance:
            return True
    return False


def get_cam_intrinsics(cam_dict):
    """
    Parameters
    ----------
    cam_dict : dict
        a dictionary containing all camera intrinsics
    Returns
    -------
    list
        relevant list of relevant cam intrinsics
    """
    height = cam_dict['Height']
    width = cam_dict['Width']
    intrinsic0 = literal_eval(cam_dict['IntrinsicMatrix.0'])
    intrinsic1 = literal_eval(cam_dict['IntrinsicMatrix.1'])

    data = [width, height, intrinsic0[0],
            intrinsic1[1], intrinsic0[2], intrinsic1[2]]

    if "LensDistortionInverseLookupTable" in cam_dict:
        inverse_lut = cam_dict["LensDistortionLookupTable"]
        dist_reference_dims = literal_eval(
            cam_dict["IntrinsicMatrixReferenceDimensions"])
        distortion_center = literal_eval(cam_dict["LensDistortionCenter"])
        # scale distortion center to intrinsics
        scaler = width / dist_reference_dims[0]
        distortion_center = np.array(distortion_center) * scaler
        return data, inverse_lut, distortion_center

    return data, None, None


def bilinear_interpolation_01(x, y, values):
    """Interpolate values given at the corners of
    [0,1]x[0,1] square.

    Parameters:
        x : float
        y : float
        points : ((v00, v01), (v10, v11))
            input grid with 4 values from which to interpolate.
            Inner dimension = x, thus v01 = value at (x=1,y=0).

    Returns:
        float
            interpolated value
    """
    return (values[0][0] * (1 - x) * (1 - y) +
            values[0][1] * x * (1 - y) +
            values[1][0] * (1 - x) * y +
            values[1][1] * x * y)


def linear_interpolation_01(x, values):
    """Interpolate values given at 0 and 1.

    Parameters:
        x : float
        y : float
        points : (v0, v1)
            values at 0 and 1

    Returns:
        float
            interpolated value
    """
    return values[0] * (1 - x) + values[1] * x


def interpolate_depth_value(point2d, depth_image):
    """ Bilinear interpolate a depth value

    Parameters
    ----------
    x : ndarray
        input 2d point
    depth_image : ndarray
        input depth image

    Returns
    -------
    float
        interpolated depth
    """
    x = point2d[1]
    y = point2d[0]
    xl = int(np.floor(x))
    xr = int(np.ceil(x))
    yl = int(np.floor(y))
    yr = int(np.ceil(y))

    if xl == xr:
        if yl == yr:
            return depth_image[xl, yl]
        else:
            values = [depth_image[xl, yl],
                      depth_image[xr, yr]]
            return linear_interpolation_01(y - yl, values)
    else:
        if yl == yr:
            values = [depth_image[xl, yl],
                      depth_image[xr, yr]]
            return linear_interpolation_01(x - xl, values)
        else:
            values = ((depth_image[xl, yl], depth_image[xr, yl]),
                      (depth_image[xl, yr], depth_image[xr, yr]))
            return bilinear_interpolation_01(x - xl, y - yl, values)


def unproject_pt_to_3d(point2d, depth_image, inv_cam_mat, bi_interp=True):
    """ Unproject a 2D point to 3D using a depth image and camera intrinsics
    Parameters
    ----------
    point2d : ndarray
        2D point to unproject to 3D
    depth_image : ndarray
        depth image used to unproject the 2D point
    inv_cam_mat : ndarray
        inverse of the camera matrix. Used to unproject point2d to a vector
    bi_interp : bool
        if point should be interpolated instead of taking the nearest neighbor
    Returns
    -------
    numpy array
        a 3D point corresponding to point2D and the depth
    """
    if bi_interp:
        depth = interpolate_depth_value(point2d, depth_image)
    else:
        x = int(np.round(point2d[1]))
        y = int(np.round(point2d[0]))
        if (x < depth_image.shape[0] and
           y < depth_image.shape[1] and
           x >= 0 and y >= 0):
            depth = depth_image[x, y]
        else:
            depth = 0.0

    impage_pt = depth * np.array([point2d[0], point2d[1], 1.0])
    return np.matmul(inv_cam_mat, impage_pt), depth


def get_square_length(file):
    """
    Parameters
    ----------
    file : str
        path to checkersize.txt. Contains square length in centimeters [cm].
    Returns
    -------
    float
        square length in meters [m].
    """
    with open(file) as f:
        return float(f.read()) * 1e-2


def create_aruco_board(dataset_path):
    """
    Parameters
    ----------
    dataset_path : str
        path to dataset also containing the checkersize.txt.
    Returns
    -------
    tuple
        termination criteria for supixel estimation
    cv2.aruco_CharucoBoard
        aruco board object
    cv2.aruco_DetectorParameters()
        aruco detector params
    cv2.aruco_Dictionary()
        aruco dictionary
    float
        square length in meters
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters_create()
    square_length = get_square_length(
        os.path.join(dataset_path, "checkersize.txt"))
    board = cv2.aruco.CharucoBoard_create(
        10, 8, square_length, square_length / 2.0, aruco_dict)

    return criteria, board, aruco_params, aruco_dict, square_length


def detect_corners(image, aruco_board, criteria, aruco_dict, aruco_params, cam_matrix=None):
    """
    Parameters
    ----------
    image : ndarray
        gray value image to detect corners on
    aruco_board : cv2.aruco_CharucoBoard
        aruco board object
    criteria : tuple
        subpixel termination criteria
    aruco_dict : cv2.aruco_Dictionary()
        aruco dictionary
    aruco_params : cv2.aruco_DetectorParameters()
        detector params
    cam_matrix : ndarray
        camera matrix 3x3
    Returns
    -------
    int
        number of detected corners
    list
        list of charuco_corners
    list
        list of charuco ids
    """
    corners, ids, _ = cv2.aruco.detectMarkers(
        image, dictionary=aruco_dict, parameters=aruco_params)

    if len(corners) > 0:
        # SUB PIXEL DETECTION
        for corner in corners:
            cv2.cornerSubPix(image, corner, winSize=(
                3, 3), zeroZone=(-1, -1), criteria=criteria)
        nr_pts, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, aruco_board, cameraMatrix=cam_matrix, distCoeffs=None, minMarkers=1)
        return nr_pts, charuco_corners, charuco_ids
    else:
        return None, None, None
