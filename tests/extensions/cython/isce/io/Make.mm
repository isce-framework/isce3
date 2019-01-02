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

all: test

test: raster

raster:
	${PYTHON} -m pytest ./raster.py

# end of file
