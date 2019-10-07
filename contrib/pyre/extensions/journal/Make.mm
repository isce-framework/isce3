# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include journal.def
# package name
PACKAGE =
# the module
MODULE = journal
# build a python extension
include std-pythonmodule.def
# use a tmp directory that knows the name of the module
PROJ_TMPDIR = $(BLD_TMPDIR)/extensions/$(PROJECT)
# point to the location of my libraries
PROJ_LCXX_LIBPATH=$(BLD_LIBDIR)
# link against these
PROJ_LIBRARIES = -ljournal
# the sources
PROJ_SRCS = \
    DeviceProxy.cc \
    exceptions.cc \
    channels.cc \
    init.cc \
    metadata.cc \
    tests.cc \

# end of file
