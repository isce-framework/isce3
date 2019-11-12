# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

include pyre.def
PACKAGE = algebra

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_CLEAN += $(TESTS)
PROJ_LIBRARIES =
LIBRARIES = $(PROJ_LIBRARIES) $(EXTERNAL_LIBS)

TESTS = bcd

#--------------------------------------------------------------------------

all: test clean


test: $(TESTS)
	./bcd


# build
%: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)


# end of file
