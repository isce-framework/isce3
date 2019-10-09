# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = patterns
# headers that are scoped by the package name
EXPORT_PKG_HEADERS = \
    Registrar.h Registrar.icc

# the tmp directory
PROJ_TMPDIR = $(BLD_TMPDIR)/${PROJECT}/lib/$(PROJECT)

# the standard targets
all: export

export:: export-package-headers

live: live-package-headers

# archiving support
zipit:
	cd $(EXPORT_ROOT); \
        zip -r $(PYRE_ZIP) include/pyre/$(PACKAGE)

# end of file
