import numpy as np
from numpy.typing import DTypeLike

import nornir_imageregistration
from nornir_imageregistration import ShapeLike


def CreateDistanceImageBruteForce(shape, dtype=None):
    """
    Create a distance image where the value at each pixel is the distance from the center of the image.
    This method is not intended to be used for more than testing as it is inefficient.
    :param shape:
    :param dtype:
    :return:
    """
    if dtype is None:
        dtype = nornir_imageregistration.default_depth_image_dtype()

    center = [shape[0] / 2.0, shape[1] / 2.0]

    x_range = np.linspace(-center[1], center[1], shape[1])
    y_range = np.linspace(-center[0], center[0], shape[0])

    x_range **= 2
    y_range **= 2

    distance = np.empty(shape, dtype=dtype)

    for i in range(0, shape[0]):
        distance[i, :] = x_range + y_range[i]

    distance = np.sqrt(distance)

    return distance


def CreateDistanceImage(shape: ShapeLike, dtype: DTypeLike | None = None):
    """Create a distance image where the value at each pixel is the distance from the center of the image.
    Distances are measured in pixels, and the distance is zero at the center pixel.  Distances are measured
    to the center of each pixel.
    """
    # TODO, this has some obvious optimizations available
    if dtype is None:
        dtype = nornir_imageregistration.default_depth_image_dtype()

    # center = [shape[0] / 2.0, shape[1] / 2.0]
    shape = np.asarray(shape, dtype=np.int64)
    is_odd_shape = np.fmod(shape, 2) > 0

    half_shape = None
    if True:
        even_shape = shape.copy()
        even_shape[is_odd_shape] -= 1
        half_shape = even_shape / 2
        half_shape = half_shape.astype(np.int64)

    y_range = None
    if not is_odd_shape[0]:
        y_range = np.linspace(0.5, half_shape[0] - 0.5, num=half_shape[0])
    else:
        half_shape[0] += 1
        y_range = np.linspace(0, half_shape[0] - 1, num=half_shape[0])

    x_range = None
    if not is_odd_shape[1]:
        x_range = np.linspace(0.5, half_shape[1] - 0.5, num=half_shape[1])
    else:
        half_shape[1] += 1
        x_range = np.linspace(0, half_shape[1] - 1, num=half_shape[1])

    x_range *= x_range
    y_range *= y_range

    distance = np.empty(half_shape, dtype=dtype)

    for i in range(0, half_shape[0]):
        distance[i, :] = x_range + y_range[i]

    distance = np.sqrt(distance)

    # OK, mirror the array as needed to build the final image
    if not is_odd_shape[1]:
        distance = np.hstack((np.fliplr(distance), distance))
    else:
        distance = np.hstack((np.fliplr(distance[:, 1:]), distance))

    if not is_odd_shape[0]:
        distance = np.vstack((np.flipud(distance), distance))
    else:
        distance = np.vstack((np.flipud(distance[1:, :]), distance))

    return distance
