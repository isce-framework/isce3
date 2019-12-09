# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# access the machinery for building shared objects
include shared/target.def
# project defaults
include pyre.def
# the name of the package
PACKAGE = grid

# the products
PROJ_SAR = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SO)
# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/${PROJECT}/lib/$(PROJECT)

# what to clean
PROJ_CLEAN += $(EXPORT_INCDIR)/$(PACKAGE) $(EXPORT_INCDIR)/$(PACKAGE).h

# what to export
# the library
EXPORT_LIBS = $(PROJ_DLL)
# the public header
EXPORT_HEADERS = \
    grid.h \
# the package headers
EXPORT_PKG_HEADERS = \
    ConstGridIterator.h ConstGridIterator.icc \
    ConstView.h ConstView.icc \
    DirectGrid.h DirectGrid.icc \
    Grid.h Grid.icc \
    GridIterator.h GridIterator.icc \
    Index.h Index.icc \
    Iterator.h Iterator.icc \
    Layout.h Layout.icc \
    Packing.h Packing.icc \
    Slice.h Slice.icc \
    View.h View.icc \
    public.h

# the standard targets
all: export

export:: export-package-headers

live: live-headers live-package-headers
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); \
        zip -r $(PYRE_ZIP) lib/lib$(PROJECT).$(EXT_SO); \
        zip -r $(PYRE_ZIP) ${addprefix include/pyre/, $(EXPORT_HEADERS)} ; \
        zip -r $(PYRE_ZIP) include/pyre/$(PACKAGE)

# end of file
