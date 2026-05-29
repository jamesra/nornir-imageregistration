import os
import pickle
from typing import Any, Callable
import unittest


class PickleHelper:
    @property
    def CachePath(self) -> str:
        """Contains cached files from previous test runs, such as database query results.
           Entries in this cache should have a low probablility of changing and breaking tests"""
        if 'TESTOUTPUTPATH' in os.environ:
            test_output_dir = os.environ["TESTOUTPUTPATH"]
            return os.path.join(test_output_dir, "Cache", self.__class__.__name__)
        else:
            raise EnvironmentError("TESTOUTPUTPATH environment variable should specify test output directory")

    @staticmethod
    def _ensure_pickle_extension(path: str) -> str:
        (_, ext) = os.path.splitext(path)
        if ext != '.pickle':
            path = os.path.join(path, '.pickle')
        return path

    def SaveVariable(self, var: Any, path: str):
        """Save the passed variable to disk as a pickle file"""
        path = PickleHelper._ensure_pickle_extension(path)

        fullpath = os.path.join(self.CachePath, path)

        if not os.path.exists(os.path.dirname(fullpath)):
            os.makedirs(os.path.dirname(fullpath))

        with open(fullpath, 'wb') as filehandle:
            print("Saving: " + fullpath)
            pickle.dump(var, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

    def ReadOrCreateVariable(self, varname: str, createfunc: Callable[..., Any] | None = None, **kwargs) -> Any:
        """Reads variable from disk. If createfunc is provided call it with the loaded variable and return the result.
           If the variable was loaded previously in the test the cached version will be used."""

        # If the variable is already defined in the test class use it instead of going to disk again
        var = None
        if hasattr(self, varname):
            var = getattr(self, varname)

        if var is None:
            path = os.path.join(self.CachePath, varname + ".pickle")
            path = PickleHelper._ensure_pickle_extension(path)
            if os.path.exists(path):
                with open(path, 'rb') as filehandle:
                    try:
                        var = pickle.load(filehandle)
                    except:
                        var = None
                        print("Unable to load graph from pickle file: " + path)

            if var is None and not createfunc is None:
                var = createfunc(**kwargs)
                self.SaveVariable(var, path)

        return var
