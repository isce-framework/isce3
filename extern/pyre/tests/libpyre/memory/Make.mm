# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def

# the pile of tests
TESTS = \
    view-instantiate \
    constview-instantiate \
    heap-instantiate \
    direct-create \
    direct-map \
    direct-instantiate \
    direct-instantiate-partial \
    constdirect-create \
    constdirect-map \
    constdirect-instantiate \
    constdirect-instantiate-partial \

# tests that should fail because their access patterns are prohibited
SHOULD_FAIL = \
    direct-clone \
    constdirect-clone \

all: test clean

# testing
test: $(TESTS)
	@echo "testing:"
	@for testcase in $(TESTS); do { \
            echo "    $${testcase}" ; ./$${testcase} || exit 1 ; \
            } done

# build
PROJ_CLEAN += $(TESTS) grid.dat
PROJ_LIBRARIES = -lpyre -ljournal
LIBRARIES = $(PROJ_LIBRARIES) $(EXTERNAL_LIBS)

%: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)

# end of file
