# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# get the cereal defaults
include cereal.def

# project global settings
PROJECT = cereal

PACKAGE = external/rapidxml

# my subdirectories
RECURSE_DIRS = \

EXPORT_PKG_HEADERS = \
    rapidxml.hpp \
    rapidxml_iterators.hpp \
    rapidxml_print.hpp \
    rapidxml_utils.hpp \


# the standard targets

all: export
	BLD_ACTION="all" $(MM) recurse

export:: export-headers export-package-headers
        BLD_ACTION="export $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse


# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
