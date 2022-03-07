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

import numpy as np

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
