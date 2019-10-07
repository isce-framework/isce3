# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# access the project defaults
include {project.name}.def
# the name of the package
PACKAGE = tests

# standard targets
all: test

test: sanity

sanity:
	${{PYTHON}} ./sanity.py
	${{PYTHON}} ./extension.py

live:

# end of file
