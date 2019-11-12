# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# access the project defaults
include {project.name}.def
# the package name
PACKAGE = bin

# export these
EXPORT_BINS = \
    {project.name} \

# add these to the clean pile
PROJ_CLEAN = ${{addprefix $(EXPORT_BINDIR)/, $(EXPORT_BINS)}}

# standard targets
all: export

export:: export-binaries

live: live-bin

# end of file
