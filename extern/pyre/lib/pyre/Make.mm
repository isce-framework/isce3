# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# access the machinery for building shared objects
include shared/target.def
# project defaults
include pyre.def
# my subdirectories
RECURSE_DIRS = \
    algebra \
    geometry \
    grid \
    memory \
    patterns \
    timers \


# the products
PROJ_SAR = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(EXT_SO)
# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/${PROJECT}/lib/$(PROJECT)
# what to clean
PROJ_CLEAN += $(EXPORT_LIBS) $(EXPORT_INCDIR)

# the sources
PROJ_SRCS = \
    version.cc \

# what to export
# the library
EXPORT_LIBS = $(PROJ_DLL)
# the top level headers
EXPORT_HEADERS = \
    geometry.h \
    grid.h \
    memory.h \
    timers.h \
    version.h

# get today's date
TODAY = ${strip ${shell date -u}}
# grab the revision number
REVISION = ${strip ${shell git log --format=format:"%h" -n 1}}
# if not there
ifeq ($(REVISION),)
REVISION = 0
endif

# the standard targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: version.cc $(PROJ_DLL) export-headers export-libraries
	BLD_ACTION="export" $(MM) recurse
	@$(RM) version.cc

revision: version.cc $(PROJ_DLL) export-libraries
	@$(RM) version.cc

live:
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	PYRE_ZIP=$(PYRE_ZIP) BLD_ACTION="zipit" $(MM) recurse

# construct my {version.cc}
version.cc: version.cc.in Make.mm
	@sed \
          -e "s:@MAJOR@:$(PROJECT_MAJOR):g" \
          -e "s:@MINOR@:$(PROJECT_MINOR):g" \
          -e "s:@MICRO@:$(PROJECT_MICRO):g" \
          -e "s:@REVISION@:$(REVISION):g" \
          -e "s|TODAY|$(TODAY)|g" \
          version.cc.in > version.cc

# end of file
