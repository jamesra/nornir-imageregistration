'''
Created on Mar 12, 2013

@author: u0490822

Converts the output from ImageMagick -histogram:info:- flag to a histogram
'''

import nornir_shared.histogram


def MinMaxValues(lines):
    '''Given output from the histogram command, return the min/max values'''
    minVal = None;
    maxVal = None;

    for line in lines:
        (intensityVal, count) = ParseHistogramLine(line);
        if not intensityVal is None:
            minVal = intensityVal;
            break;

    for iLine in range(len(lines) - 1, 0, -1):
        line = lines[iLine];
        (intensityVal, count) = ParseHistogramLine(line);
        if not intensityVal is None:
            maxVal = intensityVal;
            break;

    return (minVal, maxVal);

def ParseHistogramLine(line):
    line = line.strip();

    if(len(line) == 0):
        return (None, None)

    parts = line.split(':', 1);
    if len(parts) <= 1:
        return  (None, None)

    if len(parts[0]) <= 0:
        return  (None, None)

    count = int(parts[0])

    # Split out the pixel intensity from the portion after the colon

    iStartTuple = parts[1].find('(');
    iEndTuple = parts[1].find(')');

    if iStartTuple < 0 or iEndTuple < 0:
        return  (None, None)

    tupleStr = str(parts[iStartTuple:iEndTuple]);
    tupleParts = tupleStr.split(',');

    intensityVal = int(tupleParts[1]);

    return (intensityVal, count)


def Parse(lines, minVal=None, maxVal=None, numBins=None):
    # Each line of the convert  -define histogram:unique-colors=true -format %c histogram:info:- command should have this form:
    #    1: ( 33, 33, 33) #212121 gray(33,33,33)
    #    2: ( 35, 35, 35) #232323 gray(35,35,35)
    #    7: ( 36, 36, 36) #242424 gray(36,36,36)

    if numBins is None:
        numBins = 256;

    # Find the max/min values first
    if minVal is None or maxVal is None:
        (ActualMin, ActualMax) = MinMaxValues(lines);

        if minVal is None:
            minVal = ActualMin;

        if maxVal is None:
            maxVal = ActualMax;

    # If this happens perhaps ImageMagick doesn't sort the output anymore?
    assert(minVal < maxVal)

    if minVal is None:
        minVal = 0;

    if maxVal is None:
        maxVal = 255;

    hist = nornir_shared.histogram.Histogram.Init(minVal, maxVal, numBins)

    for line in lines:
        (intensityVal, count) = ParseHistogramLine(line);
        if not intensityVal is None:
            hist.IncrementBin(intensityVal, count);

    return hist
