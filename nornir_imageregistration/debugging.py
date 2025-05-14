import sys
import os

if 'DEBUG' in os.environ or sys.gettrace() is not None:
    try:
        value = os.environ['DEBUG']
        if value:
            value = int(value)
            __in_debug_mode = bool(value)
        else:
            __in_debug_mode = True
    except ValueError:
        print('DEBUG environment variable must be 0 or 1 or empty.  (Empty is equivalent to 1)')
        raise
else:
    __in_debug_mode = False


def in_debug_mode() -> bool:
    return __in_debug_mode
