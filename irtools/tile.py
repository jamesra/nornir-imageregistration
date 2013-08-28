'''
Created on Jul 10, 2012

@author: Jamesan
'''

class Tile(object):
    '''
    classdocs
    '''

    @property
    def X(self):
        return self.__X;

    def __init__(self, Filename, x, y, width, height):
        '''
        Constructor
        '''

        self.__X = x;
        self.__Y = y;
        self.__Width = width;
        self.__Height = height;

