# -*- Makefile -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2017 all rights reserved
#

# project defaults
include isce.def

# the pile of tests
TESTS = \
    poly1d \

all: test

test: poly1d

poly1d:
	nosetests ./poly1d.py

# end of file
