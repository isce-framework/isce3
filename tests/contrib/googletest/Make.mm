# -*- Makefile -*-
#
# eric m. gurrola
# (c) 2017 all rights reserved
#

# project defaults
include isce.def

# the pile of tests
TESTS = \
    square_root_test2 \

all: test clean

# testing
test: $(TESTS)
	@echo "testing:"
	@for testcase in $(TESTS); do { \
            echo "    $${testcase}" ; ./$${testcase} || exit 1 ; \
            } done

# build
PROJ_CLEAN += $(TESTS)
PROJ_LIBRARIES = -lisce.$(PROJECT_MAJOR).$(PROJECT_MINOR) -lpyre -ljournal -lgtest
LIBRARIES = $(PROJ_LIBRARIES) $(EXTERNAL_LIBS)

%: %.$(EXT_CXX)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)

# end of file
