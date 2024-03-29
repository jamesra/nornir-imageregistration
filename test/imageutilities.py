import numpy as np
from numpy.typing import NDArray
import nornir_imageregistration

try:
    import cupy as cp
except ModuleNotFoundError:
    import nornir_imageregistration.cupy_thunk as cp
except ImportError:
    import nornir_imageregistration.cupy_thunk as cp


def use_cp() -> bool:
    """Return true if cupy is being used"""
    return nornir_imageregistration.GetActiveComputationLib() == nornir_imageregistration.ComputationLib.cupy


def create_tiny_image(shape: nornir_imageregistration.AreaLike, num_shades: int = 4) -> NDArray[np.floating]:
    """Creates an image composed of rotating colors"""
    shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(shape, np.int32)
    image = np.zeros(shape,
                     dtype=np.float32)  # Needs to be float32 for numpy functions used in the tests to work

    for x in range(0, shape[1]):
        for y in range(0, shape[0]):
            color_index = (((x % num_shades) + (y % num_shades)) % num_shades) / num_shades
            image[y, x] = (color_index * 0.8) + 0.2  # Ensure we don't have black

    # Make origin bright white
    image[0, 0] = 1.0

    return image


def create_gradient_image(shape: nornir_imageregistration.ShapeLike, min_val: float = 0.2, max_val: float = 0.8,
                          num_shades: int = 8) -> NDArray[np.floating]:
    """

    :param shape:
    :param min: Minimum intensity of gradient
    :param max: Maximum intensity of gradient
    :param num_shades: Number of different shades in the output image
    :return:
    """
    shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(shape, np.int32)
    image = np.zeros(shape, dtype=nornir_imageregistration.default_image_dtype())
    for x in range(0, shape[1]):
        for y in range(0, shape[0]):
            color_index = (((x % num_shades) + (y % num_shades)) % num_shades) / num_shades
            image[y, x] = (color_index * max_val) + min_val

    if use_cp():
        image = cp.asarray(image)

    return image


def create_nested_squares_image(shape: nornir_imageregistration.ShapeLike, min_val: float = 0.2,
                                max_val: float = 0.8, num_shades: int = 8):
    """

    :param shape:
    :param min: Minimum intensity of gradient
    :param max: Maximum intensity of gradient
    :param num_shades: Number of different shades in the output image
    :return:
    """
    shape = nornir_imageregistration.EnsurePointsAre1DNumpyArray(shape, np.int32)
    image = np.zeros(shape, dtype=nornir_imageregistration.default_image_dtype())
    half = shape / 2.0
    half_shade_step = (1 / num_shades) / 3.0
    for x in range(0, shape[1]):
        for y in range(0, shape[0]):
            quad = tuple(np.array((x, y)) >= half)
            if quad == (True, True):
                color_index = min(shape[1] - x, shape[0] - y)
            elif quad == (False, False):
                color_index = min(x, y)
            elif quad == (True, False):
                color_index = min(shape[1] - x, y)
            else:  # quad == (False, True):
                color_index = min(x, shape[0] - y)

            color_index %= num_shades

            # color_index = (((x % num_shades) + (y % num_shades)) % num_shades)
            color_index /= num_shades

            if (x + y) % 2 == 0:
                color_index += half_shade_step

            image[y, x] = (color_index * max_val) + min_val

    if use_cp():
        image = cp.asarray(image)

    return image
