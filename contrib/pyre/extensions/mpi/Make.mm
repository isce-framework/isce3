# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include mpi.def
# package name
PACKAGE =
# the module
MODULE = mpi
# mpi support
include MPI/default.def
# buidl a python extension
include std-pythonmodule.def
# my headers
PROJ_INCDIR = $(BLD_INCDIR)/pyre/$(PROJECT)
# use a tmp directory that knows the name of the module
PROJ_TMPDIR = $(BLD_TMPDIR)/extensions/$(PROJECT)
# link against these
PROJ_LIBRARIES = -ljournal
# the sources
PROJ_SRCS = \
    communicators.cc \
    exceptions.cc \
    groups.cc \
    metadata.cc \
    ports.cc \
    startup.cc

export:: export-headers

EXPORT_INCDIR = $(EXPORT_ROOT)/include/pyre/$(PROJECT)
EXPORT_HEADERS = \
    capsules.h

# end of file
