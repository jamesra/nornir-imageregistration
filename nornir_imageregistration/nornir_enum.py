'''
Created on Jun 4, 2013

@author: u0490822
'''


def enum(*sequential, **named):
    '''Generates a dictionary of names to number values used as an enumeration.
       Copied from a StackOverflow post'''
    enums = dict(list(zip(sequential, list(range(len(sequential))))), **named)
    return type('Enum', (), enums)
