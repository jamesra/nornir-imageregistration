import hypothesis
import hypothesis.strategies
import hypothesis.extra.numpy
import numpy as np
from nornir_imageregistration.spatial import Rectangle


@hypothesis.strategies.composite
def rectangles(draw, minCount=0, maxCount=128, shapeRange=None, positionRange=None):
    if positionRange is None:
        positionRange = (-100, 100)

    if shapeRange is None:
        shapeRange = (
            0,
            (positionRange[1] - positionRange[0]) + 1)  # Default to a max shape that spans the entire range of origins

    nRects = draw(hypothesis.strategies.integers(min_value=minCount, max_value=maxCount))
    points = draw(hypothesis.extra.numpy.arrays(np.float32,
                                                shape=(nRects, 2),
                                                elements=hypothesis.strategies.floats(positionRange[0],
                                                                                      positionRange[1], width=16),
                                                fill=hypothesis.strategies.nothing()))
    shapes = draw(hypothesis.extra.numpy.arrays(np.uint8,
                                                shape=(nRects, 2),
                                                elements=hypothesis.strategies.integers(shapeRange[0], shapeRange[1]),
                                                fill=hypothesis.strategies.nothing()))

    rect_list = []
    for i in range(points.shape[0]):
        origin = points[i, :]
        shape = shapes[i, :]

        rect_list.append(Rectangle.CreateFromPointAndArea(origin, shape))

    return rect_list
