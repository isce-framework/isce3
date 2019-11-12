# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# access the project defaults
include {project.name}.def

# adjust the build parameters
PROJ_LIB = $(PROJ_LIBDIR)/lib{project.name}.$(EXT_LIB)
# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/{project.name}/lib/{project.name}

# the list of sources to compile
PROJ_SRCS = \
    version.cc

# the public headers
EXPORT_HEADERS = \
    version.h

# the library
EXPORT_LIBS = $(PROJ_LIB)

# the stuff to clean up
PROJ_CLEAN += \
    $(PROJ_INCDIR) \
    $(PROJ_LIB) \
    $(PROJ_TMPDIR) \
    $(EXPORT_INCDIR) \
    $(EXPORT_LIBDIR)/lib{project.name}.$(EXT_LIB)

# the default target compiles this library and exports it
all: export-headers proj-lib export-libraries

live: live-headers live-libraries

# end of file
