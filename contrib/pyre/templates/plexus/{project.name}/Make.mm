# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# access the project defaults
include {project.name}.def
# the package name
PACKAGE = {project.name}
# clean up
PROJ_CLEAN += $(EXPORT_MODULEDIR)

# the list of directories to visit
RECURSE_DIRS = \
    actions \
    components \
    extensions \

# the list of python modules
EXPORT_PYTHON_MODULES = \
    meta.py \
    __init__.py

# get today's date
TODAY = ${{strip ${{shell date -u}}}}
# grab the revision number
REVISION = ${{strip ${{shell git log --format=format:"%h" -n 1}}}}
# if not there
ifeq ($(REVISION),)
REVISION = 0
endif

# the standard build targets
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

# construct my {{meta.py}}
meta.py: meta Make.mm
	@sed \
          -e "s:MAJOR:$(PROJECT_MAJOR):g" \
          -e "s:MINOR:$(PROJECT_MINOR):g" \
          -e "s:REVISION:$(REVISION):g" \
          -e "s|TODAY|$(TODAY)|g" \
          meta > meta.py


# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
