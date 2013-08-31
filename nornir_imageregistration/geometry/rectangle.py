'''
Created on Oct 25, 2012

@author: u0490822
'''

class Rectangle(object):
    '''
    classdocs
    '''

    @property
    def width(self):
        return self.bounds[2] - self.bounds[0];

    @property
    def height(self):
        return self.bounds[3] - self.bounds[1];

    def __init__(self, bounds):
        '''
        Constructor, bounds = [left bottom right top]
        '''

        self.bounds = bounds;

    @classmethod
    def contains(cls, rect, primitive):
        '''If len == 2 primitive is a point,
           if len == 4 primitive is a rect [left bottom right top]'''
        IsPoint = len(primitive) == 2;
        IsRect = len(primitive) == 4;

        if(IsPoint):
            point = primitive;
            if point[0] < rect[0] or \
                point[0] > rect[2] or \
                point[1] < rect[1] or \
                point[1] > rect[3]:
                    return False;

            return True;
        if(IsRect):
            if(primitive[2] < rect[0] or
                   primitive[0] > rect[2] or
                   primitive[3] < rect[1] or
                   primitive[1] > rect[3]):
                    return False;
            return True;
