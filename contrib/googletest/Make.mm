# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
PROJECT = gtest
# my subdirectories
RECURSE_DIRS = \
    include \
    src \

# the standard targets

all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse


# shortcuts for building specific subdirectories
.PHONY: include src

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
