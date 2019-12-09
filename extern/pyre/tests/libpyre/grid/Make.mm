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
LAYOUT_TESTS = \
    index \
    index-access \
    index-bool \
    index-arithmetic \
    packing \
    packing-c \
    packing-fortran \
    packing-access \
    slice \
    iterator \
    iterator-access \
    iterator-loop \
    iterator-slice \
    layout \
    layout-order \
    layout-order-default \
    layout-slice \

GRID_TESTS = \
    grid-view \
    grid-heap \
    grid-direct \
    grid-direct-data \
    grid-direct-set \
    grid-direct-get \
    grid-fill \
    grid-mosaic \
    grid-scan \
    grid-transform \
    grid-view-assignment \

all: test clean

# testing
test: layout_tests grid_tests

layout_tests: $(LAYOUT_TESTS)
	@echo "testing:"
	@for testcase in $(LAYOUT_TESTS); do { \
            echo "    $${testcase}" ; ./$${testcase} || exit 1 ; \
            } done

grid_tests: $(GRID_TESTS)
	@echo "testing:"
	@for testcase in $(GRID_TESTS); do { \
            echo "    $${testcase}" ; ./$${testcase} || exit 1 ; \
            } done

# build
PROJ_CLEAN += $(LAYOUT_TESTS) $(GRID_TESTS) grid.dat
PROJ_LIBRARIES = -lpyre -ljournal
LIBRARIES = $(PROJ_LIBRARIES) $(EXTERNAL_LIBS)

%: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)

# end of file
