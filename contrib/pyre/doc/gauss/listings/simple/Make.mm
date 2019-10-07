# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

PROJECT = pyre
PACKAGE = doc/gauss/simple

PROJ_TIDY += gauss gauss.dSYM __pycache__

include gsl/default.def

#--------------------------------------------------------------------------
#

all: gauss test clean

gauss: gauss.cc
	$(CXX) $(CXXFLAGS) -o $@ gauss.cc $(LCXXFLAGS) $(EXTERNAL_LIBS)

test: gauss
	./gauss
	$(PYTHON) gauss.py

# end of file
