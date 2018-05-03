#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

# Define a helper function to convert Python strings/bytes to bytes
def pyStringToBytes(s):
    if isinstance(s, str):
        return s.encode('utf-8')
    elif isinstance(s, bytes):
        return s
    else:
        return s

def dummy():
    pass

include "pyTopo.pyx"
