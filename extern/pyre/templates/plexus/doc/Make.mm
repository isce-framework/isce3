# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# access the project defaults
include {project.name}.def
# the package name
PACKAGE = doc

# the pile of things to clean
PROJ_CLEAN += \
    $(PROJ_DOCDIR)

all: tidy

live:

# end of file
