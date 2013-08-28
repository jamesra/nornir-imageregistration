'''
Created on Jul 10, 2012

@author: Jamesan
'''


class Layout(object):
    '''
    Records a set of images and their positions
    '''
    @property
    def OverlapMap(self):
        if(self.__overlapMap is None):
            self.__overlapMap = Layout.CalculateOverlapMap(self.dictImages);

        return self.__overlapMap;


    def __init__(self, dictImages):
        '''
        dictImages should be a dictionary of the form: 
        dictImages[ImageFileName] = [x y width height]
        '''

        self.dictImages = dictImages;
        self.__overlapMap = None;


    @classmethod
    def CalculateOverlapMap(cls, dictImages):
        '''dictImages[ImageFileName] = [x y width height]'''
        '''returns sparse matrix with 1 set for overlapping tiles'''
