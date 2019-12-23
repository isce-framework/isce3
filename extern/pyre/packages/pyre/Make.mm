# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the name of the package
PACKAGE = pyre
# add this to the clean pile
PROJ_CLEAN += $(EXPORT_MODULEDIR)
# my subfolders
RECURSE_DIRS = \
    algebraic \
    calc \
    components \
    config \
    constraints \
    db \
    descriptors \
    extensions \
    externals \
    filesystem \
    flow \
    framework \
    geometry \
    grid \
    handbook \
    http \
    ipc \
    nexus \
    parsing \
    patterns \
    platforms \
    primitives \
    records \
    schemata \
    shells \
    tabular \
    timers \
    tracking \
    traits \
    units \
    weaver \
    xml

# the python modules
EXPORT_PYTHON_MODULES = \
    meta.py \
    services.py \
    __init__.py

# get today's date
TODAY = ${strip ${shell date -u}}
# grab the revision number
REVISION = ${strip ${shell git log --format=format:"%h" -n 1}}
# if not there
ifeq ($(REVISION),)
REVISION = 0
endif

# standard targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: meta.py export-python-modules
	BLD_ACTION="export" $(MM) recurse
	@$(RM) meta.py

live: live-python-modules
	BLD_ACTION="live" $(MM) recurse

revision: meta.py export-python-modules
	@$(RM) meta.py

# construct my {meta.py}
meta.py: meta.py.in Make.mm
	@sed \
          -e "s:@MAJOR@:$(PROJECT_MAJOR):g" \
          -e "s:@MINOR@:$(PROJECT_MINOR):g" \
          -e "s:@MICRO@:$(PROJECT_MICRO):g" \
          -e "s:@REVISION@:$(REVISION):g" \
          -e "s|@TODAY@|$(TODAY)|g" \
          meta.py.in > meta.py

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
