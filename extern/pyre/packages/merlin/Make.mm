# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project defaults
include merlin.def
# package name
PACKAGE = merlin
# my subfolders
RECURSE_DIRS = \
    assets \
    components \
    schema \
    spells \
# the python modules
EXPORT_PYTHON_MODULES = \
    meta.py \
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

# construct my {meta.py}
meta.py: meta.py.in Make.mm
	@sed \
          -e "s:@MAJOR@:$(PROJECT_MAJOR):g" \
          -e "s:@MINOR@:$(PROJECT_MINOR):g" \
          -e "s:@MICRO@:$(PROJECT_MICRO):g" \
          -e "s:@REVISION@:$(REVISION):g" \
          -e "s|@TODAY@|$(TODAY)|g" \
          meta.py.in > meta.py

# end of file
