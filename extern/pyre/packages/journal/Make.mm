# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include journal.def
# package name
PACKAGE = journal
# the python modules
EXPORT_PYTHON_MODULES = \
    ANSIRenderer.py \
    Channel.py \
    Console.py \
    Debug.py \
    Device.py \
    Diagnostic.py \
    Error.py \
    File.py \
    Firewall.py \
    Info.py \
    Journal.py \
    Renderer.py \
    TextRenderer.py \
    Warning.py \
    exceptions.py \
    meta.py \
    protocols.py \
    proxies.py \
    schemes.py \
    __init__.py

# get today's date
TODAY = ${strip ${shell date -u}}
# grab the revision number
REVISION = ${strip ${shell git log --format=format:"%h" -n 1}}
# if not there
ifeq ($(REVISION),)
REVISION = 0
endif

# standard targets
all: export

export:: meta.py export-python-modules
	@$(RM) meta.py

live: live-python-modules

# construct my {meta.py}
meta.py: meta.py.in Make.mm
	@sed \
          -e "s:@MAJOR@:$(PROJECT_MAJOR):g" \
          -e "s:@MINOR@:$(PROJECT_MINOR):g" \
          -e "s:@MICRO@:$(PROJECT_MICRO):g" \
          -e "s:@REVISION@:$(REVISION):g" \
          -e "s|@TODAY@|$(TODAY)|g" \
          meta.py.in > meta.py

# end of file
