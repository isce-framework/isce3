# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package name
PACKAGE = timers/posix
# headers scoped by the package name
EXPORT_PKG_HEADERS = \
    Clock.h Clock.icc

# the tmp directory
PROJ_TMPDIR = $(BLD_TMPDIR)/${PROJECT}/lib/$(PROJECT)

# the standard targets
all: export

export:: export-package-headers

live: live-package-headers

# end of file
