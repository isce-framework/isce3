# -*- Makefile -*-
#
# Bryan V. Riel
# (c) 2017 all rights reserved
#

# project defaults
include isce.def

# the pile of tests
TESTS = \
    topo \

all: test clean

# testing
test: $(TESTS)
	@echo "testing:"
	@for testcase in $(TESTS); do { \
            echo "    $${testcase}" ; \
            ./$${testcase} || exit 1 ; \
            } done

# build
PROJ_CLEAN += $(TESTS) lat.rdr lat.hdr lon.rdr lon.hdr z.rdr z.hdr inc.rdr inc.hdr \
              hdg.rdr hdg.hdr localInc.rdr localInc.hdr localPsi.rdr localPsi.hdr \
              simamp.rdr simamp.hdr
PROJ_CXX_INCLUDES += $(EXPORT_ROOT)/include/$(PROJECT)-$(PROJECT_MAJOR).$(PROJECT_MINOR)
PROJ_LIBRARIES = -lisce.$(PROJECT_MAJOR).$(PROJECT_MINOR) -lgtest
LIBRARIES = $(PROJ_LIBRARIES) $(EXTERNAL_LIBS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)

# end of file
