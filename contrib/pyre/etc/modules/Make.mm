# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project globals
include pyre.def

# the module files
PROJ_MODULES = \
    pyre \

# get today's date
TODAY = ${strip ${shell date -u}}
# grab the revision number
REVISION = ${strip ${shell git log --format=format:"%h" -n 1}}
# if not there
ifeq ($(REVISION),)
REVISION = 0
endif

# grab the repo nickname
NICKNAME = ${strip ${shell git rev-parse --abbrev-ref HEAD}}

# the standard targets
all: export-modules

# pyre
pyre: $(EXPORT_MODDIR) Make.mm
	@sed \
          -e "s|TODAY|$(TODAY)|g" \
          -e "s:PYRE_MAJOR:$(PROJECT_MAJOR):g" \
          -e "s:PYRE_MINOR:$(PROJECT_MINOR):g" \
          -e "s:EXPORT_ROOT:$(EXPORT_ROOT):g" \
          -e "s:EXPORT_BINDIR:$(EXPORT_BINDIR):g" \
          -e "s:EXPORT_LIBDIR:$(EXPORT_LIBDIR):g" \
          -e "s:EXPORT_INCDIR:$(EXPORT_INCDIR):g" \
          -e "s:EXPORT_PKGDIR:$(EXPORT_PKGDIR):g" \
          pyre > $(EXPORT_MODDIR)/$(NICKNAME)

.PHONY: $(PROJ_MODULES)

# end of file
