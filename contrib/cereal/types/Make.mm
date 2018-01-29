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

PACKAGE = types

# my subdirectories
RECURSE_DIRS = \
    concepts \

EXPORT_PKG_HEADERS = \
    array.hpp \
    base_class.hpp \
    bitset.hpp \
    boost_variant.hpp \
    chrono.hpp \
    common.hpp \
    complex.hpp \
    deque.hpp \
    forward_list.hpp \
    functional.hpp \
    list.hpp \
    map.hpp \
    memory.hpp \
    polymorphic.hpp \
    queue.hpp \
    set.hpp \
    stack.hpp \
    string.hpp \
    tuple.hpp \
    unordered_map.hpp \
    unordered_set.hpp \
    utility.hpp \
    valarray.hpp \
    vector.hpp \


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
