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

PACKAGE = details

# my subdirectories
RECURSE_DIRS = \

EXPORT_PKG_HEADERS = \
    helpers.hpp \
    polymorphic_impl.hpp \
    polymorphic_impl_fwd.hpp \
    static_object.hpp \
    traits.hpp \
    util.hpp \

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
