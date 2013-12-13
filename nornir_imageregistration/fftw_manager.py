'''
Created on Jul 12, 2012

@author: Jamesan
'''

import threading

import numpy

import fftw3f as fftw3


class FFTWPlan(object):

    def __init__(self, fft, ifft, InputArray, OutputArray):
        self.__fft = fft;
        self.__ifft = ifft;
        self.InputArray = InputArray;
        self.OutputArray = OutputArray;

    def fft(self, Image):
        if Image.shape != self.InputArray.shape:
            raise ValueError("FFT Plan dimensions do not match input image");

        self.InputArray[:] = Image;
        self.__fft();
        return self.OutputArray.copy();

    def ifft(self, Image):
        if Image.shape != self.InputArray.shape:
            raise ValueError("FFT Plan dimensions do not match input image");

        self.OutputArray[:] = Image;
        self.__ifft();
        return self.InputArray.copy();


class FFTWManager(object):
    '''
    Performs fft and ifft using the pyFFTW library.
    '''
    tls = threading.local();

    def __init__(self):
        '''
        Constructor
        '''

        self.dictPlans = dict();


    @classmethod
    def GetFFTManager(cls):
        # Use thread local storage to find the plan manager if it exists

        cThread = threading.current_thread();
        if not hasattr(cls.tls, 'FFTWMan'):
            cls.tls.FFTWMan = FFTWManager();
            cls.tls.cThread = cThread;


        return cls.tls.FFTWMan;


    def GetPlan(self, SizeTuple):
        if SizeTuple in self.dictPlans:
            return self.dictPlans[SizeTuple];

        InputArray = fftw3.create_aligned_array(SizeTuple, dtype = numpy.singlecomplex);
        OutputArray = fftw3.create_aligned_array(SizeTuple, dtype = numpy.singlecomplex);

        fft = fftw3.Plan(InputArray, OutputArray, direction = 'forward', flags = ['measure']);
        ifft = fftw3.Plan(OutputArray, InputArray, direction = 'backward', flags = ['measure']);

        PlanObj = FFTWPlan(fft, ifft, InputArray, OutputArray);
        self.dictPlans[SizeTuple] = PlanObj;

        return PlanObj;


if __name__ == '__main__':

    FFTMan = FFTWManager();
    PlanObj = FFTMan.GetPlan((256, 256));


    import TaskTimer;
