# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
include isce.def

# my subdirectories
RECURSE_DIRS = \
    $(PACKAGES)

# the ones that are always available
PACKAGES = \
    core \
    geometry \

# the products
PROJ_DLL = $(BLD_LIBDIR)/lib$(PROJECT).$(PROJECT_MAJOR).$(PROJECT_MINOR).$(EXT_SO)
# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)-$(PROJECT_MAJOR).$(PROJECT_MINOR)/lib
# what to clean
PROJ_CLEAN += $(EXPORT_LIBS) $(EXPORT_INCDIR)

# what to export
# the library
EXPORT_LIBS = $(PROJ_DLL)

# project settings: do not remove core directory (core usually refers core dump file)
# filter-out info at: https://www.gnu.org/software/make/manual/html_node/index.html
PROJ_TIDY := ${filter-out core, $(PROJ_TIDY)}

# the standard targets

all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); zip -r $(PYRE_ZIP) ${addprefix packages/, $(PACKAGES) --include \*.py}

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
