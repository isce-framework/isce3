# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package name
PACKAGE = bin
# the list of files
EXPORT_BINS = \
    {project.name}

# add these to the clean pile
PROJ_CLEAN = ${{addprefix $(EXPORT_BINDIR)/, $(EXPORT_BINS)}}

# the standard build targets
all: export

export:: export-binaries

live: live-bin

# end of file
