#!/usr/bin/env python3

from isce3.extensions.isceextension import pyRaster
import numpy

err = None

def test_raster_filenotfound():
    """
    Test resulting exception when a raster is created from a nonexistent file
    """

    global err

    try:
        # Open a nonexistent file
        pyRaster("nonexistent_filename")

        # We shouldn't be here (exception should be raised)
        assert(False)

    except Exception as e:
        err = e

        # Extract error info
        filename = e.args[0][0] # which C++ file the error was raised from
        lineno   = e.args[0][1] # which line number in that file
        funcname = e.args[0][2] # the name of the C++ function
        errmsg = e.args[1] # a string containing all of the above,
                           # as a readymade printable error message

        # check types
        assert(type(filename) is str)
        assert(type(lineno)   is int)
        assert(type(funcname) is str)

        # check error message
        assert(filename    in errmsg)
        assert(str(lineno) in errmsg)
        assert(funcname    in errmsg)

        # check values
        assert("Raster.cpp" in filename)
        assert("Raster("    in funcname)

test_raster_filenotfound()
