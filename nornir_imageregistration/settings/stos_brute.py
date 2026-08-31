import enum

from numpy.typing import NDArray
import numpy as np
from typing import NamedTuple, Sequence, Iterable, AbstractSet
from pydantic import BaseModel, ConfigDict
from nornir_imageregistration.settings.angle_range import AngleSearchRange


class SliceToSliceMethod(enum.Enum):
    BruteForce = 1  # Test every angle in a list of provided angles and choose the best
    LogPolar = 2  # Use log polar registration to find the best angle and scale in one calculation


class StosBruteSettings(BaseModel):
    """Encodes the settings required or used to invoke StosBrute"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    angles: AngleSearchRange | Sequence[float] | AbstractSet[float] | None = None
    min_overlap: float = 0.75  # The minimum amount of overlap we require in the images.  Higher values reduce false positives but may not register offset images
    source_image_scale_factors: tuple[float, ...] | NDArray[np.floating] | None = None
    """Amount to scale the warped image before attempting registration,
    this handles cases where multiple scopes are used with slightly differnt magnification values """

    larget_dimension: int | None = None  # The input images should be scaled so the largest image dimension is equal to this value, default is 1024.  None means use the actual image size
    try_flipped: bool = True  # If True the algorithm will test the flipped version of the source image too
    estimated_scale_hint: float | None = None
    """Optional isotropic scale hint from log-polar on raw images (total scale; converted to residual internally)."""
    initial_scale_hint: float | None = None
    """Caller-provided total scale (e.g. current transform scalar) to seed scale search/refinement."""
    search_scale: bool = True
    """If False, register at scale 1.0: no log-polar scale probe, no scale search or refinement,
    making the search a pure translation (plus the angle range) match.  Default True preserves
    the scale-searching behaviour every existing caller relies on."""
    _method: SliceToSliceMethod

    @property
    def method(self) -> SliceToSliceMethod:
        return self._method

    @method.setter
    def method(self, value: SliceToSliceMethod):
        self._method = value

    def __init__(self,
                 method: SliceToSliceMethod | None = None,
                 angles: AngleSearchRange | Sequence[float] | None = None,
                 min_overlap: float = 0.75,
                 source_image_scale_factors: NDArray[np.floating] | None = None,
                 larget_dimension: int | None = 1024,
                 try_flipped: bool = True,
                 estimated_scale_hint: float | None = None,
                 initial_scale_hint: float | None = None,
                 search_scale: bool = True,
                 ):
        """
        :param angles: Angles to search for the best control point alignment or None if all angles should be searched
        :param min_overlap: Minimum amount of overlap we require to consider a registration to be valiud
        :param source_image_scale_factors:  Amount to scale the warped image before attempting registration, this handles cases where multiple scopes are used with slightly differnt magnification values
        :param larget_dimension: The input images should be scaled so the largest image dimension is equal to this value, default is 1024.  None means use the actual image size
        :param try_flipped: If True the algorithm will test the flipped version of the source image too
        :param search_scale: If False, register at scale 1.0 with no scale probe, search or refinement
        """
        super().__init__()
        self._method = method  # type: ignore[assignment]
        self.angles = angles

        self.min_overlap = min_overlap
        self.larget_dimension = larget_dimension
        self.try_flipped = try_flipped
        self.estimated_scale_hint = estimated_scale_hint
        self.initial_scale_hint = initial_scale_hint
        self.search_scale = search_scale
        self._method = SliceToSliceMethod.LogPolar if method is None else method

        self.source_image_scale_factors = source_image_scale_factors

        if self.source_image_scale_factors is None:
            pass
        elif isinstance(source_image_scale_factors, np.ndarray):
            self.source_image_scale_factors = source_image_scale_factors
        elif hasattr(source_image_scale_factors, '__iter__'):
            self.source_image_scale_factors = np.array(source_image_scale_factors, float)
        else:
            self.source_image_scale_factors = np.array([source_image_scale_factors, source_image_scale_factors], float)

    def angle_range_defined(self) -> bool:
        return self.angles is not None

    @property
    def angle_range(self) -> NDArray[np.floating]:
        """:return: The range of angles to search for the best control point alignment or None if all angles should be searched"""
        if self.angles is None:
            return np.array(range(-178, 182, 2), float)

        if isinstance(self.angles, np.ndarray):
            return self.angles
        elif isinstance(self.angles, AngleSearchRange):
            return self.angles.angle_range
        elif isinstance(self.angles, Iterable):
            return np.array(list(self.angles), float)

        raise ValueError(f"Unexpected type for self.angle_search_settings: {self.angles.__class__}")

    @property
    def source_image_scaling_required(self) -> bool:
        if self.source_image_scale_factors is None:
            return False

        return bool(np.any(self.source_image_scale_factors != 1))

