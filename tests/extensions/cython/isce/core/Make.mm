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
	${PYTHON} -m pytest ./attitude.py

ellipsoid:
	${PYTHON} -m pytest ./ellipsoid.py

orbit:
	${PYTHON} -m pytest ./orbit.py

poly1d:
	${PYTHON} -m pytest ./poly1d.py

# end of file
