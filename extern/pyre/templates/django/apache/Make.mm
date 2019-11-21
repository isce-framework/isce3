# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package name
PACKAGE = web/apache
# the apache configuration files
PROJ_LIVE_APACHE_CONF = $(PROJECT).conf
PROJ_LIVE_APACHE_WSGI = $(PROJECT).wsgi
# assemble the entire set
PROJ_LIVE_APACHE_ALL = $(PROJ_LIVE_APACHE_CONF) $(PROJ_LIVE_APACHE_WSGI)

# the standard build targets
all: tidy

live: live-apache-conf live-apache-restart

# end of file
