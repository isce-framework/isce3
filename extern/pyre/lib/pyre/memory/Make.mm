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
PACKAGE = memory

# the products
PROJ_SAR = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SO)
# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/${PROJECT}/lib/$(PROJECT)
# the sources
PROJ_SRCS = \
    MemoryMap.cc \

# what to clean
PROJ_CLEAN += $(EXPORT_INCDIR)/$(PACKAGE) $(EXPORT_INCDIR)/$(PACKAGE).h

# what to export
# the library
EXPORT_LIBS = $(PROJ_DLL)
# the public header
EXPORT_HEADERS = \
    memory.h \
# the package headers
EXPORT_PKG_HEADERS = \
    ConstDirect.h ConstDirect.icc \
    ConstView.h ConstView.icc \
    Direct.h Direct.icc \
    Heap.h Heap.icc \
    MemoryMap.h MemoryMap.icc \
    View.h View.icc \
    forward.h public.h

# the standard targets
all: export

export:: $(PROJ_DLL) export-package-headers export-libraries

live: live-headers live-package-headers live-libraries
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); \
        zip -r $(PYRE_ZIP) lib/lib$(PROJECT).$(EXT_SO); \
        zip -r $(PYRE_ZIP) ${addprefix include/pyre/, $(EXPORT_HEADERS)} ; \
        zip -r $(PYRE_ZIP) include/pyre/$(PACKAGE)

# end of file
