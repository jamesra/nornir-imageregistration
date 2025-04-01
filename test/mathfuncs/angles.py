import numpy as np
import unittest


def are_angle_radians_equal(angle1, angle2, tolerance=1e-5):
    # Normalize angles to the range [0, 2*pi)
    angle1 = np.mod(angle1, 2 * np.pi)
    angle2 = np.mod(angle2, 2 * np.pi)
    return np.abs(angle1 - angle2) < tolerance or np.abs(angle1 - angle2 - (2 * np.pi)) < tolerance


def are_angle_degrees_equal(angle1, angle2, tolerance=1e-5):
    # Normalize angles to the range [0, 2*pi)
    angle1 = np.mod(angle1, 360)
    angle2 = np.mod(angle2, 360)
    return np.abs(angle1 - angle2) < tolerance or np.abs(angle1 - angle2 - 360) < tolerance


def assert_angles_equal(obj: unittest.TestCase, angle1, angle2, tolerance=1e-5, msg: str | None = None):
    if not are_angle_radians_equal(angle1, angle2, tolerance):
        if msg is None:
            msg = ""
        obj.fail(f"Angles are not equal: {angle1} != {angle2} : {msg}")


def assert_angles_equal_degrees(obj: unittest.TestCase, angle1, angle2, tolerance=1e-5, msg: str | None = None):
    if not are_angle_degrees_equal(angle1, angle2, tolerance):
        if msg is None:
            msg = ""
        obj.fail(f"Angles are not equal: {angle1} != {angle2} : {msg}")
