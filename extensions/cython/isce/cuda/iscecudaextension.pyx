#cython: language_level=3
#
# Author: Bryan V. Riel, Joshua Cohen
# Copyright 2017
#

# Define a helper function to convert Python strings/bytes to bytes
def pyStringToBytes(s):
    if isinstance(s, str):
        return s.encode('utf-8')
    elif isinstance(s, bytes):
        return s
    else:
        raise ValueError('Input Python string not str or bytes')

# Include the geometry extensions
include "geometry/pyTopo.pyx"
include "geometry/pyGeo2rdr.pyx"


# end of file
