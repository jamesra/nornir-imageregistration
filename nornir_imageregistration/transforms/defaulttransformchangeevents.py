import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from abc import ABCMeta
from typing import List, Callable

from nornir_imageregistration.transforms.base import ITransformChangeEvents
import nornir_pools


class DefaultTransformChangeEvents(ITransformChangeEvents, metaclass=ABCMeta):
    # Static thread pool executor
    _executor = ThreadPoolExecutor()
    OnChangeEventListeners: List[Callable[[], None]]

    def __init__(self):
        self.OnChangeEventListeners = []

    def __getstate__(self):
        # odict = super(GridWithRBFFallback, self).__getstate__()
        odict = dict()
        return odict

    def __setstate__(self, dictionary):
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
            # with ThreadPoolExecutor as executor:
            tlist = [DefaultTransformChangeEvents._executor.submit(func) for func in self.OnChangeEventListeners]
            for task in concurrent.futures.as_completed(tlist):
                # Call result to see exceptions
                result = task.result()

            # pool = nornir_pools.GetGlobalThreadPool()
            # tlist = list()
            #
            # for func in self.OnChangeEventListeners:
            #     tlist.append(pool.add_task("OnTransformChanged calling " + str(func), func))
            #
            # # Call wait on all tasks so we see exceptions
            # while len(tlist) > 0:
            #     t = tlist.pop(0)
            #     t.wait()
        else:
            for func in self.OnChangeEventListeners:
                func()
