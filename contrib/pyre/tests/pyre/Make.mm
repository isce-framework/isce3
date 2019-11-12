# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

RECURSE_DIRS = \
    primitives \
    patterns \
    grid \
    parsing \
    units \
    filesystem \
    xml \
    schemata \
    constraints \
    algebraic \
    calc \
    descriptors \
    records \
    tabular \
    tracking \
    codecs \
    config \
    framework \
    components \
    timers \
    weaver \
    db \
    ipc \
    nexus \
    platforms \
    shells \
    externals \
    flow \
    pyre

#
all:
	BLD_ACTION="all" $(MM) recurse

test::
	BLD_ACTION="test" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))

# end of file
