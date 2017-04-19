'''
Created on Oct 18, 2012

@author: Jamesan
'''

import nornir_pools as pools


class Base(object):
    '''Base class of all transforms'''

    def __init__(self):
        self.OnChangeEventListeners = []

    @classmethod
    def Load(cls, TransformString):
        pass

    def Transform(self, point, **kwargs):
        return None

    def InverseTransform(self, point, **kwargs):
        return None

    def AddOnChangeEventListener(self, func):
        self.OnChangeEventListeners.append(func)

    def RemoveOnChangeEventListener(self, func):
        if func in self.OnChangeEventListeners:
            self.OnChangeEventListeners.remove(func)

    ThreadPool = None

    def OnTransformChanged(self):
        '''Calls every function registered to be notified when the transform changes.'''

        # Calls every listener when the transform has changed in a way that a point may be mapped to a new position in the fixed space        
        
        
        if len(self.OnChangeEventListeners) > 1:
            Pool = pools.GetGlobalThreadPool() 
            tlist = list()
                    
            for func in self.OnChangeEventListeners:
                tlist.append(Pool.add_task("OnTransformChanged calling " + str(func), func))
    
            Pool.wait_completion()
        else:
            for func in self.OnChangeEventListeners:
                func()
