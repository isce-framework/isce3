# -*- Makefile -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2017 all rights reserved
#

# project defaults
include isce.def

# the pile of tests
TESTS = \
    attitude \
    ellipsoid \
    orbit \
    poly1d \

all: test

test: attitude ellipsoid orbit poly1d

attitude:
	nosetests ./attitude.py

ellipsoid:
	nosetests ./ellipsoid.py

orbit:
	nosetests ./orbit.py

poly1d:
	nosetests ./poly1d.py

# end of file
