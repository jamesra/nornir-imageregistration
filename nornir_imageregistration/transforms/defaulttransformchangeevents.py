from abc import ABCMeta

import nornir_pools
from nornir_imageregistration.transforms.base import ITransformChangeEvents


class DefaultTransformChangeEvents(ITransformChangeEvents, metaclass=ABCMeta):
    def __init__(self):
        self.OnChangeEventListeners = []

    def AddOnChangeEventListener(self, func):
        self.OnChangeEventListeners.append(func)

    def RemoveOnChangeEventListener(self, func):
        if func in self.OnChangeEventListeners:
            self.OnChangeEventListeners.remove(func)

    def OnTransformChanged(self):
        """Calls every function registered to be notified when the transform changes."""

        # Calls every listener when the transform has changed in a way that a point may be mapped to a new position in the fixed space

        if len(self.OnChangeEventListeners) > 1:
            pool = nornir_pools.GetGlobalThreadPool()
            tlist = list()

            for func in self.OnChangeEventListeners:
                tlist.append(pool.add_task("OnTransformChanged calling " + str(func), func))

            # Call wait on all tasks so we see exceptions
            while len(tlist) > 0:
                t = tlist.pop(0)
                t.wait()
        else:
            for func in self.OnChangeEventListeners:
                func()
