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
# the package name
PACKAGE = timers
# my subfolders
RECURSE_DIRS = \
    epoch \
    mach \
    posix \

# libraries
PROJ_SAR = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SO)
PROJ_TMPDIR = $(BLD_TMPDIR)/${PROJECT}/lib/$(PROJECT)
# sources
PROJ_SRCS = \
    Display.cc \
    Timer.cc \
# the target
EXPORT_LIBS = $(PROJ_DLL)
# headers scoped by the package name
EXPORT_PKG_HEADERS = \
    Display.h Display.icc \
    Timer.h Timer.icc

# standard targets
all: export
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: $(PROJ_DLL) export-package-headers export-libraries
	BLD_ACTION="export" $(MM) recurse

live: live-headers live-package-headers live-libraries
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); \
        zip -r $(PYRE_ZIP) lib/lib$(PROJECT).$(EXT_SO); \
        zip -r $(PYRE_ZIP) ${addprefix include/pyre/, $(EXPORT_HEADERS)} ; \
        zip -r $(PYRE_ZIP) include/pyre/$(PACKAGE)

# end of file
