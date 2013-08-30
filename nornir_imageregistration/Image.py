'''
Created on Oct 4, 2012

@author: u0490822
'''


from pylab import *


class Image(object):
    '''
    Class used for images, caches calculated stats
    '''

    @property
    def Width(self):
        return Image.shape[0];

    @property
    def Height(self):
        return Image.shape[1];

    @property
    def Data(self):
        return self._ImData;

    @property
    def Median(self):
        if not hasattr(self, '_Median'):
            self.CalcMedianStdDevStats();

        return self._Median;

    @property
    def StdDev(self):
        if not hasattr(self, '_StdDev'):
            self.CalcMedianStdDevStats();

        return self._StdDev;


    def CalcMedianStdDevStats(self):
        Image1D = reshape(self.Data, self.Width * self.Height, 1);
        self._Median = median(Image1D);
        self._StdDev = std(Image1D);


    def __init__(self, Image):
        '''
        Constructor
        '''

        self._ImData = Image;






