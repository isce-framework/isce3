# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package
PACKAGE = extensions
# the module
MODULE = {project.name}
# build a python extension
include std-pythonmodule.def

# adjust the build parameters
PROJ_LCXX_LIBPATH=$(BLD_LIBDIR)
# the list of extension source files
PROJ_SRCS = \
    exceptions.cc \
    metadata.cc

# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/{project.name}/extensions/{project.name}
# my dependencies
PROJ_LIBRARIES += -l{project.name} -ljournal

# register the dependence on {{lib{project.name}}} so I get recompiled when it changes
PROJ_OTHER_DEPENDENCIES = $(BLD_LIBDIR)/lib{project.name}.$(EXT_AR)

# the pile of things to clean
PROJ_CLEAN += \
    $(PROJ_CXX_LIB) \
    $(MODULE_DLL) \
    $(EXPORT_BINDIR)/$(MODULE).abi3.$(EXT_SO)

# end of file
