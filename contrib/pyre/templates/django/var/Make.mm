# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package name
PACKAGE = var
# the stuff in this directory goes to {{var/{project.name}}}
EXPORT_ETCDIR = $(EXPORT_ROOT)/$(PACKAGE)/$(PROJECT)

# the standard build targets
all: export

export:: export-etcdir

live: live-vardir

# end of file
