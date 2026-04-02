import sys
import os

__in_debug_mode = False
__debug_mode_checked = False

def _check_debug_mode_environment_variable() -> bool:
    if sys.gettrace() is not None:
        return True
    
    if 'DEBUG' in os.environ:
        print('DEBUG environment variable is deprecated. Use DEBUG instead.')
        value = os.environ['DEBUG']
        if value:
            try:
                value = int(value)
                return bool(value)
            except ValueError:
                print('DEBUG environment variable must be 0 or 1 or empty.  (Empty is equivalent to 1)')
                raise
        else:
            return True
            
    return False



def in_debug_mode() -> bool:
    global __in_debug_mode
    global __debug_mode_checked
    if __debug_mode_checked:
        return __in_debug_mode
    
    __debug_mode_checked = True
    __in_debug_mode = _check_debug_mode_environment_variable()
    
    return __in_debug_mode
 
