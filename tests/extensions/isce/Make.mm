# -*- Makefile -*-
#
# eric m. gurrola
# jet propulsion laboratory
# california institute of technology
# (c) 2013 all rights reserved
#


PROJECT = spisce

#--------------------------------------------------------------------------
#

all: test

test: estimate_dop

estimate_dop:
	${PYTHON} ./estimate_dop.py

# end of file 
