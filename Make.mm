# -*- Makefile -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 2012-2013 all rights reserved
#


# get the global defaults
include spisce.def

# the list of directories to visit
RECURSE_DIRS = \
    packages \
    lib \
    extensions \
    defaults \
    bin \
    tests \
    examples \
    doc \
#    developers \

#--------------------------------------------------------------------------
# the recursive targets

all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export::
	BLD_ACTION="export" $(MM) recurse

release::
	BLD_ACTION="release" $(MM) recurse

#--------------------------------------------------------------------------
#  shortcuts to building in subdirectories

.PHONY: bin defaults doc examples tests

bin:
	(cd bin; $(MM))

defaults:
	(cd defaults; $(MM))

doc:
	(cd doc; $(MM))

examples:
	(cd examples; $(MM))

extensions:
	(cd extensions; $(MM))

packages:
	(cd packages; $(MM))

tests:
	(cd tests; $(MM))

users:
	(cd users; $(MM))


# end of file
