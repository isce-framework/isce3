# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package name
PACKAGE = defaults
# my subdirectories
RECURSE_DIRS = \
    actions

# the configuration files
EXPORT_ETC = \
    {project.name}.pfg

# add these to the clean pile
PROJ_CLEAN += ${{addprefix $(EXPORT_ETCDIR)/, $(EXPORT_ETC)}}

# the standard targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-etc
	BLD_ACTION="export" $(MM) recurse

live: live-etc
	BLD_ACTION="live" $(MM) recurse

# end of file
