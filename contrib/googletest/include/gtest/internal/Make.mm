# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
PROJECT = gtest

# the package name
PACKAGE = internal

# my subdirectories
RECURSE_DIRS =  custom

# the top level headers
EXPORT_PKG_HEADERS = \
        gtest-death-test-internal.h \
        gtest-filepath.h \
        gtest-internal.h \
        gtest-linked_ptr.h \
        gtest-param-util-generated.h \
        gtest-param-util.h \
        gtest-port-arch.h \
        gtest-port.h \
        gtest-string.h \
        gtest-tuple.h \
        gtest-type-util.h \

all: export
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse

export:: export-headers export-package-headers
        BLD_ACTION="export $(MM) recurse

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
