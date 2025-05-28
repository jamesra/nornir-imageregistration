import sys
import os

__in_debug_mode = False

if 'DEBUG' in os.environ or sys.gettrace() is not None:
    try:
        if 'DEBUG' in os.environ:
            print('DEBUG environment variable is deprecated. Use DEBUG instead.')
            value = os.environ['DEBUG']
            if value:
                value = int(value)
                __in_debug_mode = bool(value)
            else:
                __in_debug_mode = True
        else:
            __in_debug_mode = True
    except ValueError:
        print('DEBUG environment variable must be 0 or 1 or empty.  (Empty is equivalent to 1)')
        raise


def in_debug_mode() -> bool:
    global __in_debug_mode
    return __in_debug_mode
