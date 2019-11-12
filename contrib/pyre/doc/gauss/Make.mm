# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

PROJECT = pyre
PACKAGE = doc/gauss

RECURSE_DIRS = \
   config \
   diagrams \
   figures \
   listings \
   sections \

DOCUMENT = gauss

INCLUDES = \
    config/*.sty \
    config/*.tex \

SECTIONS = \
    $(DOCUMENT).tex \
    sections/*.tex \

FIGURES = \
    figures/*.pdf \

LISTINGS = \
    listings/simple/gauss.py \
    listings/simple/gauss.cc \
    listings/classes/*.py \
    listings/containers/*.py \
    listings/generators/*.py \
    ../../examples/gauss.pyre/gauss/*.py \
    ../../examples/gauss.pyre/gauss/functors/*.py \
    ../../examples/gauss.pyre/gauss/integrators/*.py \
    ../../examples/gauss.pyre/gauss/meshes/*.py \
    ../../examples/gauss.pyre/gauss/shapes/*.py \

#
all: $(DOCUMENT).pdf

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

test:
	BLD_ACTION="all" $(MM) recurse

# preview types
osx: $(DOCUMENT).pdf
	open $(DOCUMENT).pdf

xpdf: $(DOCUMENT).pdf
	xpdf -remote $(DOCUMENT) $(DOCUMENT).pdf

# make the document using the default document class
$(DOCUMENT).pdf: $(DOCUMENT).tex $(INCLUDES) $(SECTIONS) $(FIGURES) $(LISTINGS)

# housekeeping
PROJ_CLEAN += $(CLEAN_LATEX) *.snm *.nav *.vrb
PROJ_DISTCLEAN = *.ps *.pdf $(PROJ_CLEAN)

# end of file
