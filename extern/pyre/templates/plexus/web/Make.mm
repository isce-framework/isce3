# -*- Makefile -*-
#
# authors:
#   {project.authors}
#
# (c) {project.span} all rights reserved
#

# project settings
include {project.name}.def
# the package
PACKAGE = web
# my subdirectories
RECURSE_DIRS = \
    fonts \
    graphics \
    scripts \
    styles \


# layout
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_WEBPACK_CONFIG = config
PROJ_WEBPACK_SOURCES = react
PROJ_CLEAN += ${{addprefix $(PROJ_TMPDIR)/, build $(PROJ_WEBPACK_SOURCES)}}

# the exported items
EXPORT_WEB = \
   $(PROJ_TMPDIR)/build/*

# standard targets
all: webpack.deps webpack.build export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-web
	BLD_ACTION="export" $(MM) recurse

live: live-dirs live-web
	BLD_ACTION="live" $(MM) recurse

# end of file
