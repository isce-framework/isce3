# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = weaver
# the python modules
EXPORT_PYTHON_MODULES = \
    Banner.py \
    BlockComments.py \
    BlockMill.py \
    C.py \
    CSh.py \
    Cfg.py \
    Cxx.py \
    Django.py \
    Expression.py \
    F77.py \
    F90.py \
    HTML.py \
    HTTP.py \
    Host.py \
    Indenter.py \
    Installation.py \
    Language.py \
    LineComments.py \
    LineMill.py \
    Make.py \
    Mill.py \
    MixedComments.py \
    PFG.py \
    Perl.py \
    Plexus.py \
    Project.py \
    ProjectTemplate.py \
    Python.py \
    SQL.py \
    SVG.py \
    Smith.py \
    Sh.py \
    Stationery.py \
    TeX.py \
    Weaver.py \
    XML.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
