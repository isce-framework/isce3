# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# access the project defaults
include {project.name}.def
# the package name
PACKAGE = base
# my subdirectories
RECURSE_DIRS = \
    migrations \
# the python modules
EXPORT_PYTHON_MODULES = \
    admin.py \
    models.py \
    tests.py \
    urls.py \
    views.py \
    __init__.py

# the standard build targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-package-python-modules
	BLD_ACTION="export" $(MM) recurse

live: live-package-python-modules
	BLD_ACTION="live" $(MM) recurse


# end of file
