'''
Created on Oct 18, 2012

@author: Jamesan
'''

import nornir_pools
from abc import ABC, ABCMeta, abstractproperty, abstractmethod, abstractstaticmethod, abstractclassmethod


class Base(ABC):
    '''Base class of all transforms'''

    def __init__(self):
        self.OnChangeEventListeners = []

    @classmethod
    def Load(cls, TransformString):
        pass
    
    @abstractmethod
    def Transform(self, point, **kwargs):
        '''Map points from the mapped space to fixed space. Nornir is gradually transitioning to a source space to target space naming convention.'''
        return None
    
    @abstractmethod
    def InverseTransform(self, point, **kwargs):
        '''Map points from the fixed space to mapped space. Nornir is gradually transitioning to a target space to source space naming convention.'''
        return None
    
    @property
    def MappedBoundingBox(self):
        raise NotImplementedError()
 
    @property
    def FixedBoundingBox(self):
        raise NotImplementedError()
    
    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        raise NotImplementedError()

    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        raise NotImplementedError()
    
    def AddOnChangeEventListener(self, func):
        self.OnChangeEventListeners.append(func)

    def RemoveOnChangeEventListener(self, func):
        if func in self.OnChangeEventListeners:
            self.OnChangeEventListeners.remove(func)
         
    def OnTransformChanged(self):
        '''Calls every function registered to be notified when the transform changes.'''

        # Calls every listener when the transform has changed in a way that a point may be mapped to a new position in the fixed space        

        if len(self.OnChangeEventListeners) > 1:
            Pool = nornir_pools.GetGlobalThreadPool() 
            tlist = list()

            for func in self.OnChangeEventListeners:
                tlist.append(Pool.add_task("OnTransformChanged calling " + str(func), func))
    
            #Call wait on all tasks so we see exceptions
            while len(tlist) > 0:
                t = tlist.pop(0)
                t.wait()
        else:
            for func in self.OnChangeEventListeners:
                func()
                
class ITransformTranslation(ABC):
    @abstractmethod
    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        raise NotImplementedError()

    @abstractmethod
    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        raise NotImplementedError()
    
