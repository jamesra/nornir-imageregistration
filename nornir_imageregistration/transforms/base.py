'''
Created on Oct 18, 2012

@author: Jamesan
'''

import nornir_pools
from abc import ABC, ABCMeta, abstractproperty, abstractmethod, abstractstaticmethod, abstractclassmethod

class ITransform(ABC):
     
    @abstractmethod
    def Transform(self, point, **kwargs):
        '''Map points from the mapped space to fixed space. Nornir is gradually transitioning to a source space to target space naming convention.'''
        pass
    
    @abstractmethod
    def InverseTransform(self, point, **kwargs):
        '''Map points from the fixed space to mapped space. Nornir is gradually transitioning to a target space to source space naming convention.'''
        pass
      
    @abstractclassmethod
    def Load(cls, TransformString, pixelSpacing=None):
        '''
        Creates an instance of the transform from the TransformString
        '''
        pass
    
class ITransformChangeEvents(ABC):
    @abstractmethod
    def AddOnChangeEventListener(self, func):
        '''
        Call func whenever the transform changes
        '''
        pass

    @abstractmethod
    def RemoveOnChangeEventListener(self, func):
        '''
        Stop calling func whenever the transform changes
        '''
        pass
    
class ITransformTranslation(ABC):
    @abstractmethod
    def TranslateFixed(self, offset):
        '''Translate all fixed points by the specified amount'''
        raise NotImplementedError()

    @abstractmethod
    def TranslateWarped(self, offset):
        '''Translate all warped points by the specified amount'''
        raise NotImplementedError()
    
class ITransformScaling(ABC):
    @abstractmethod
    def ScaleFixed(self, offset):
        '''Scale all fixed points by the specified amount'''
        raise NotImplementedError()

    @abstractmethod
    def ScaleWarped(self, offset):
        '''Scale all warped points by the specified amount'''
        raise NotImplementedError()
    
class IDiscreteTransform(ITransform):
    @abstractproperty
    def MappedBoundingBox(self):
        '''Bounding box of mapped space points'''
        raise NotImplementedError()
 
    @abstractproperty
    def FixedBoundingBox(self):
        '''Bounding box of fixed space points'''
        raise NotImplementedError()

class DefaultTransformChangeEvents(ITransformChangeEvents, ABC):
    def __init__(self):
        self.OnChangeEventListeners = []
        
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
    
class Base(ITransform, ITransformTranslation):
    '''Base class of all transforms'''
    pass

    
       
    
                
    
