# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package name
PACKAGE = defaults/{project.name}/actions

# actions
EXPORT_ETC = \
    debug.py

# add these to the clean pile
PROJ_CLEAN += $(EXPORT_ETCDIR)

# the standard build targets
all: export

export:: export-etc

live: live-etc

# end of file
