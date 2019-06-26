# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
PROJECT = gtest

# the package name
PACKAGE =

# my subdirectories
RECURSE_DIRS = internal

# the top level headers
EXPORT_HEADERS = \
    gtest.h \
    gtest-death-test.h \
    gtest-matchers.h \
    gtest-message.h \
    gtest-param-test.h \
    gtest-printers.h \
    gtest-spi.h \
    gtest-test-part.h \
    gtest-typed-test.h \
    gtest_pred_impl.h \
    gtest_prod.h \

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

export:: export-headers
        BLD_ACTION="export" $(MM) recurse

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
