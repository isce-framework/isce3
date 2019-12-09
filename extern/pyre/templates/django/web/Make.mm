# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project settings
include {project.name}.def
# the package
PACKAGE=web/www

# the target folders
EXPORT_RESOURCES = resources
EXPORT_TEMPLATES = templates
# the package
EXPORT_WEB = $(EXPORT_RESOURCES) $(EXPORT_TEMPLATES)
# dependency management
PROJ_LIVE_BOWER = /usr/local/bin/bower install

# standard targets
all: export

export:: export-web

live: live-web
	$(SSH) $(PROJ_LIVE_USERURL) \
           '$(CD) $(PROJ_LIVE_PCKGDIR)/$(PROJECT)/$(EXPORT_RESOURCES); $(PROJ_LIVE_BOWER)'

# end of file
