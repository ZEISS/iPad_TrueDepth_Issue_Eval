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
import natsort
import glob
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (get_cam_intrinsics,
                   create_aruco_board, detect_corners)
from depth_utils import unproject_pt_to_3d

# This script implements the procedure from section 4.2 of the paper


def run_eval_depth(path, dataset, plot_debug=True, use_own_calib_data=False):
    dataset_path = os.path.join(path, dataset)
    print("Evaluating: {}.".format(dataset))
    image_names = natsort.natsorted(
        glob.glob(os.path.join(dataset_path, "rgb_*.png")))
    depth_names = natsort.natsorted(
        glob.glob(os.path.join(dataset_path, "depth_*.png")))

    criteria, board, aruco_params, aruco_dict, sq_length = create_aruco_board(
        dataset_path)

    with open(os.path.join(dataset_path, "DepthMetadata.json"), 'r') as f:
        calib_dict = json.load(f)

    # if you want to use own calibration data you need to run calibrate_front_camera.py first
    if use_own_calib_data:
        # get phone version
        with open(os.path.join("FrontCamCalibrationDataset", dataset, "cam_calibration.json"), 'r') as f:
            calib = json.load(f)
        cam_matrix = np.array(calib["cam_matrix"])
        dist = np.array(calib["dist"])
    else:  # use factory intrinsics
        intrinsics, inverse_lut, _ = get_cam_intrinsics(calib_dict[0])

        cam_matrix = np.array(((intrinsics[2], 0.0, intrinsics[4]),
                               (0.0, intrinsics[3], intrinsics[5]),
                               (0.0, 0.0, 1.0)))
        # distortion is not explicitly available with factory intrinsics.
        # It's encoded in the inverse_lut
        dist = None
    # to unproject depth we need the inverse camera matrix. Section 2.3 eq. (2)
    inv_cam_matrix = np.linalg.inv(cam_matrix)

    results = {}
    # refine
    for idx, i_name in enumerate(image_names):
        # STEP 1
        I = cv2.imread(i_name, 0)
        # depth image png. scale by 1e5 to map to meter
        D = cv2.imread(depth_names[idx], -1) / 10000.

        # STEP 2: extract charuco corners
        nr_pts, charuco_corners, charuco_ids = detect_corners(
            I, board, criteria, aruco_dict, aruco_params, cam_matrix)
        if not nr_pts:
            continue

        if nr_pts > 4:
            objPts = board.chessboardCorners[charuco_ids, :]
            # STEP 3.1: estimate camera pose
            ret, R_c_w, t_c_w = cv2.solvePnP(
                objPts, charuco_corners, cam_matrix,
                dist, flags=cv2.SOLVEPNP_ITERATIVE)
            R_c_w = cv2.Rodrigues(R_c_w)[0]

            # get 3d point from sensor
            depth_sensor = []
            depth_board = []
            board_3d_pt = []
            arkit_3d_pt = []
            # STEP 3 and 4: - unproject depth and from depth image
            #               - transform charuco board points to camera coordinate system
            for id, pt in enumerate(charuco_corners):
                # project
                pt3d, depth = unproject_pt_to_3d(
                    pt.squeeze(), D, inv_cam_matrix, True)
                pt3d_board = R_c_w @ objPts[id, :].T + t_c_w
                depth_sensor.append(depth)
                depth_board.append(pt3d_board[2])
                board_3d_pt.append(objPts[id, :])
                arkit_3d_pt.append(pt3d)

            results[os.path.basename(i_name)] = {"ids": charuco_ids,
                                                 "corners": charuco_corners,
                                                 "depth_sensor": np.array(depth_sensor),
                                                 "depth_board": np.array(depth_board),
                                                 "board_3d_pt": np.array(board_3d_pt),
                                                 "arkit_3d_pt": np.array(arkit_3d_pt)}
            # if plot_debug:
            #     I_rgb = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
            #     I_rgb = cv2.aruco.drawAxis(
            # I_rgb, cam_matrix, None, R_c_w, t_c_w, 4*sq_length)
            #     I_rgb = cv2.aruco.drawDetectedCornersCharuco(
            #         I_rgb,  charuco_corners, charuco_ids,(255,255,0))
            #     cv2.imshow("image", I_rgb)
            #     cv2.waitKey(1)
        else:
            results[os.path.basename(i_name)] = {
                "ids": None, "corners": None, "depth_sensor": np.array([]),  "depth_board": np.array([])}

    max_mm = 20.0

    if use_own_calib_data:
        result_figure_path = os.path.join(
            dataset_path, "result_plots_own_calib")
    else:
        result_figure_path = os.path.join(
            dataset_path, "result_plots_factory_calib")
    if not os.path.exists(result_figure_path):
        os.makedirs(result_figure_path)

    # collect errors
    total_med_err = 0
    total_med_err_cnt = 0
    all_distance_errors = []
    for image in results:
        if len(results[image]["depth_sensor"]) == 0:
            continue
        # get measured sensor depth and "ground truth" charuco depth
        # and convert them to [mm]
        sensordepth = results[image]["depth_sensor"].squeeze() * 1e3
        boarddepth = results[image]["depth_board"].squeeze() * 1e3

        sensor3d = results[image]["arkit_3d_pt"].squeeze() * 1e3
        board3d_gt = results[image]["board_3d_pt"].squeeze() * 1e3

        # we could also evaluate the distance between
        # charuco points and unprojected depth points -> results are similar
        # distances_error = []
        # for i in range(board3d_gt.shape[0]):
        #     dist_gt = cv2.norm(board3d_gt[i,:] - board3d_gt[0,:])
        #     dist_sensor = cv2.norm(sensor3d[i,:] - sensor3d[0,:])
        #     distances_error.append(cv2.norm(dist_gt - dist_sensor))
        # all_distance_errors.extend(distances_error)
        # distances_error = np.array(distances_error)
        # corr_idxs = np.abs(distances_error) < 30.0
        # distances_error = distances_error[corr_idxs]

        valid_depth = sensordepth > 0

        # calculate z difference for valid depth values
        error_z = boarddepth[valid_depth] - sensordepth[valid_depth]
        # remove very outliers
        corr_idxs = np.abs(error_z) < max_mm
        error_z = error_z[corr_idxs]
        if len(error_z) == 0:
            continue
        median_err = np.median(error_z)
        total_med_err += median_err
        total_med_err_cnt += 1
        print("Median depth error for image{} : {:.2f}mm".format(image, median_err))
       #  print("Median point distance error for image{} : {:.2f}mm".format(image, np.median(distances_error)))

        sensordepth = sensordepth[sensordepth > 0]
        if plot_debug:
            fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
            # fig.tight_layout()
            sns.histplot(sensordepth, kde=True, bins=np.arange(
                170, 230, 1), color='blue', ax=axes)
            sns.histplot(boarddepth, kde=True, bins=np.arange(
                170, 230, 1), color='red', ax=axes)
            axes.grid(True)
            plt.vlines(200, 0, 20)
            axes.legend(["AVSession Raw Depth", "Aruco depth"])
            axes.set_title(
                "Distance of sensor to Aruco board. Median error: {:.2f}mm".format(median_err))
            axes.set_xlabel("Z coordinate of points [mm]")
            axes.set_ylabel("Amount of values")
            axes.set_xlim(170, 230)

            plt.savefig(os.path.join(result_figure_path,
                        "error_plot_depth_"+image+".svg"))
            plt.cla()
            plt.clf()
            plt.close()

    if total_med_err_cnt != 0:
        print("Mean median depth error: {:.2f}mm".format(
            total_med_err/total_med_err_cnt))
        # print("Median point distance error: {:.2f}mm".format(np.median(all_distance_errors)))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_all_datasets",
        help="path to all calibration datasets",
        type=str,
        default="DepthEvalDataset")
    parser.add_argument(
        "--use_own_calibration",
        help="to enable distortion calibration (not advised)",
        action="store_true")
    parser.add_argument(
        "--plot_debug",
        help="To enable printing debug images",
        action="store_true")
    args = parser.parse_args()

    calib_datasets = [name for name in os.listdir(args.path_to_all_datasets) if os.path.isdir(
        os.path.join(args.path_to_all_datasets, name))]
    for dataset in calib_datasets:
        run_eval_depth(args.path_to_all_datasets,
                       dataset,
                       plot_debug=args.plot_debug,
                       use_own_calib_data=args.use_own_calibration)
