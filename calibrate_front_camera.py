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

import cv2
import glob
import natsort
import numpy as np
import os
import json

from utils import (get_cam_intrinsics,
                   create_aruco_board, detect_corners,
                   check_if_pose_is_close)

MIN_CONERS = 8
GRID_EDGE_LENGTH = 0.04


def run_calib_front_cam(dataset_path, draw_debug=True, calib_distortion=False):

    print("Calibrating: {}".format(dataset_path))
    image_names = natsort.natsorted(
        glob.glob(os.path.join(dataset_path, "rgb_*.png")))

    criteria, board, aruco_params, aruco_dict, _ = create_aruco_board(
        dataset_path)

    # read depth metadata
    with open(os.path.join(dataset_path, "DepthMetadata.json"), 'r') as f:
        calib_dict = json.load(f)
    intrinsics = get_cam_intrinsics(calib_dict[0])[0]
    cam_matrix = np.array(((intrinsics[2], 0.0, intrinsics[4]),
                           (0.0, intrinsics[3], intrinsics[5]),
                           (0.0, 0.0, 1.0)))

    grid_poses = []

    results = {}
    # refine
    all_corners = []
    all_ids = []
    all_objpts = []
    total_nr_corners = 0
    for i_name in image_names:

        I = cv2.imread(i_name, 0)

        nr_pts, charuco_corners, charuco_ids = detect_corners(
            I, board, criteria, aruco_dict, aruco_params, cam_matrix)
        if not nr_pts:
            continue

        # check if enough corners were detected
        if nr_pts > MIN_CONERS:
            objPts = board.chessboardCorners[charuco_ids, :]
            ret, R_c_w, t_c_w = cv2.solvePnP(
                objPts, charuco_corners, cam_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ret:
                continue

            R_c_w = cv2.Rodrigues(R_c_w)[0]
            p_w_c = -R_c_w.T @ t_c_w

            # check that camera poses are not too close
            # for one dataset we must decrease the value because recording was
            # not optimal
            if check_if_pose_is_close(grid_poses, p_w_c, GRID_EDGE_LENGTH):
                continue

            grid_poses.append(p_w_c)

            total_nr_corners += len(charuco_corners)
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            all_objpts.append(board.chessboardCorners[charuco_ids, :])
            results[i_name] = {"ids": charuco_ids, "corners": charuco_corners}
            if draw_debug:
                I_rgb = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
                I_rgb = cv2.aruco.drawDetectedCornersCharuco(
                    I_rgb, charuco_corners, charuco_ids, (255, 255, 0))

                cv2.imshow("image", I_rgb)
                cv2.waitKey(1)
        else:
            results[i_name] = {"ids": None, "corners": None}

    image_size = (I.shape[1], I.shape[0])
    print("Detected a total number of {} corners in {} images".format(
        total_nr_corners, len(all_corners)))

    if calib_distortion:
        flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
    else:
        flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | \
            cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
    #
    repro_error, K_calib, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, cam_matrix, np.zeros(5), None, None, flags)

    print("Calibrated focal length: {:.2f}px".format(K_calib[0, 0]))
    print("Factory focal length: {:.2f}px".format(intrinsics[2]))
    print("Distortion coefficients: ", dist)
    print("Reprojection error: {:.2f}px".format(repro_error))
    print("Difference to factory calibration: {:.2f}% \n".format(
        (K_calib[0, 0] - intrinsics[2]) / intrinsics[2] * 100.0))

    with open(os.path.join(dataset_path, "cam_calibration.json"), 'w') as f:
        calib = {'cam_matrix': K_calib.tolist(), 'dist': dist.tolist()}
        json.dump(calib, f)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_all_datasets",
        help="path to all calibration datasets",
        type=str,
        default="FrontCamCalibrationDataset")
    parser.add_argument(
        "--calib_distortion",
        help="to enable distortion calibration (not advised)",
        action="store_true")
    parser.add_argument(
        "--plot_debug",
        help="To enable printing debug images",
        action="store_true")
    args = parser.parse_args()

    # calibrate all datasets
    calib_datasets = [
        name for name in os.listdir(args.path_to_all_datasets) if os.path.isdir(os.path.join(args.path_to_all_datasets, name))]
    for dataset in calib_datasets:
        run_calib_front_cam(os.path.join(args.path_to_all_datasets, dataset),
                            draw_debug=args.plot_debug,
                            calib_distortion=args.calib_distortion)
