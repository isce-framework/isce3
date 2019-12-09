# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# get mpi
include MPI/default.def
# the project defaults
include pyre.def
# the package name
PACKAGE = mpi
# the tmp directory
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/lib/$(PACKAGE)
# top level header
EXPORT_HEADERS = \
    mpi.h \
# headers scoped by the package name
EXPORT_PKG_HEADERS = \
    Communicator.h Communicator.icc \
    Error.h Error.icc \
    Group.h Group.icc \
    Handle.h Handle.icc \
    Shareable.h Shareable.icc \

# standard targets
all: export

export:: export-headers export-package-headers

live: live-headers live-package-headers

# archiving support
zipit:
	cd $(EXPORT_ROOT); \
        zip -r $(PYRE_ZIP) ${addprefix include/pyre/, $(EXPORT_HEADERS)} ; \
        zip -r $(PYRE_ZIP) include/pyre/$(PACKAGE)

# end of file
