# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

PROJECT = pyre
PACKAGE = overview

RECURSE_DIRS = \
   config \
   figures \
   listings \
   sections \

OTHERS = \

DOCUMENT = overview

PACKAGES =

INCLUDES = \
    config/*.sty \
    config/*.tex \

SECTIONS = \
    sections/*.tex \
    sections/*.bib \

LISTINGS = \
    listings/*.py \
    ../../examples/gauss.pyre/gauss/*.py \
    ../../examples/gauss.pyre/gauss/functors/*.py \
    ../../examples/gauss.pyre/gauss/integrators/*.py \
    ../../examples/gauss.pyre/gauss/meshes/*.py \
    ../../examples/gauss.pyre/gauss/shapes/*.py \

FIGURES = \
    figures/*.pdf \

#
all: $(DOCUMENT).pdf

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse


# preview types
osx: $(DOCUMENT).pdf
	open $(DOCUMENT).pdf

xpdf: $(DOCUMENT).pdf
	xpdf -remote $(DOCUMENT) $(DOCUMENT).pdf

# make the document using the default document class
$(DOCUMENT).pdf: $(DOCUMENT).tex $(PACKAGES) $(INCLUDES) $(SECTIONS) $(LISTINGS) $(FIGURES)

# housekeeping
PROJ_CLEAN += $(CLEAN_LATEX) *.snm *.nav *.vrb
PROJ_DISTCLEAN = *.ps *.pdf $(PROJ_CLEAN)

# end of file
