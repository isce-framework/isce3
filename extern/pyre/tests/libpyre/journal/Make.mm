# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

include journal.def
PACKAGE = journal

PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_CLEAN += $(TESTS) $(SPECIAL_TESTS)


TESTS = \
    sanity \
    chronicler \
    inventory \
    diagnostic \
    diagnostic-injection \
    channel \
    index \
    index-inventory \
    debug \
    debug-injection \
    debug-null \
    firewall \
    firewall-injection \
    firewall-null \
    info \
    info-injection \
    warning \
    warning-injection \
    error \
    error-injection \
    debuginfo \
    firewalls \

SPECIAL_TESTS = \
    debug-envvar \

PROJ_LCC_LIBPATH = $(PROJ_LIBDIR)
PROJ_LCXX_LIBPATH = $(PROJ_LIBDIR)
PROJ_LIBRARIES = -ljournal
LIBRARIES = $(PROJ_LIBRARIES) $(EXTERNAL_LIBS)

#--------------------------------------------------------------------------

all: test clean

test: $(TESTS) $(SPECIAL_TESTS)
	@echo "testing:"
	@for testcase in $(TESTS); do { echo "    $${testcase}" ; ./$${testcase} ; } done
	@echo "    debug-envvar"; DEBUG_OPT=pyre.journal.test ./debug-envvar

# build
%: %.c
	$(CC) $(CFLAGS) $^ -o $@ $(LCFLAGS) $(LIBRARIES)

%: %.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LCXXFLAGS) $(LIBRARIES)

# end of file
