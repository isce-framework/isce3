# -*- Makefile -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2017 all rights reserved
#

# project defaults
include isce.def

# the pile of tests
TESTS = \
    raster \
    h5file \

all: test

test: raster h5file

raster:
	${PYTHON} -m pytest ./raster.py

h5file:
	${PYTHON} -m pytest ./h5file.py

# end of file
