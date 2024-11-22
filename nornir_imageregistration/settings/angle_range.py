from numpy.typing import NDArray
import numpy as np
from pydantic import BaseModel
from math import pi


class AngleSearchRange(BaseModel):
    max_angle: float | None = None  # Maximum +/- deflection angle to rotate images when searching for the best control point alignment, if None, a full circle is searched at step-size intervals
    angle_step_size: float = 3  # Number of degrees to step between search angles

    @property
    def angle_range(self) -> NDArray[float]:
        if self.max_angle is None:
            angles = np.arange(start=-180, stop=180, step=self.angle_step_size)
        else:
            angles = np.arange(start=-self.max_angle,
                               stop=self.max_angle + self.angle_step_size,
                               step=self.angle_step_size)  # numpy.linspace(-7.5, 7.5, 11)

        angles = np.union1d(angles, [0])
        return angles

    def __iter__(self):
        return iter(self.angle_range)
