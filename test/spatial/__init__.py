
import hypothesis
import hypothesis.strategies
import hypothesis.extra.numpy
import numpy as np
from nornir_imageregistration import Rectangle

@hypothesis.strategies.composite
def rectangles(draw, minCount=0, maxCount=128):
    nRects = draw(hypothesis.strategies.integers(min_value=minCount, max_value=maxCount))
    points = draw(hypothesis.extra.numpy.arrays(np.float32, 
                    shape=(nRects,2), 
                    elements=hypothesis.strategies.floats(-100.0, 100.0, width=16), fill=hypothesis.strategies.nothing()))
    shapes = draw(hypothesis.extra.numpy.arrays(np.uint8,
                    shape=(nRects,2),
                    elements=hypothesis.strategies.integers(0, 255), fill=hypothesis.strategies.nothing()))
                                                
    rect_list = []
    for i in range(points.shape[0]):
        origin = points[i,:]
        shape = shapes[i,:]
        
        rect_list.append(Rectangle.CreateFromPointAndArea(origin, shape))
    
    return rect_list   
    
    