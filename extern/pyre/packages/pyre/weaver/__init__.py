# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package contains the machinery necessary to generate content in a variety of output formats.

The primary target is human readable formats, such source code for programming languages.
"""


# the marker of component factories
from .. import foundry

# access to the main components in this package
from .Weaver import Weaver as weaver
# the language interface
from .Language import Language as language

# the mill base classes
@foundry(implements=language)
def mill():
    """
    The base mill component
    """
    # grab the component class
    from .Mill import Mill as mill
    # and return it
    return mill

@foundry(implements=language)
def line():
    """
    The base mill component
    """
    # grab the component class
    from .LineMill import LineMill as line
    # and return it
    return line

@foundry(implements=language)
def block():
    """
    The base mill component
    """
    # grab the component class
    from .BlockMill import BlockMill as block
    # and return it
    return block


# access to the known languages
@foundry(implements=language)
def c():
    """
    The C weaver
    """
    # grab the component class
    from .C import C as c
    # and return it
    return c

@foundry(implements=language)
def csh():
    """
    The csh weaver
    """
    # grab the component class
    from .CSh import CSh as csh
    # and return it
    return csh

@foundry(implements=language)
def cfg():
    """
    The cfg weaver
    """
    # grab the component class
    from .Cfg import Cfg as cfg
    # and return it
    return cfg

@foundry(implements=language)
def cxx():
    """
    The C++ weaver
    """
    # grab the component class
    from .Cxx import Cxx as cxx
    # and return it
    return cxx

@foundry(implements=language)
def f77():
    """
    The FORTRAN weaver
    """
    # grab the component class
    from .F77 import F77 as f77
    # and return it
    return f77

@foundry(implements=language)
def f90():
    """
    The F90 weaver
    """
    # grab the component class
    from .F90 import F90 as f90
    # and return it
    return f90

@foundry(implements=language)
def html():
    """
    The HTML weaver
    """
    # grab the component class
    from .HTML import HTML as html
    return html

@foundry(implements=language)
def http():
    """
    The HTTP weaver
    """
    # grab the component class
    from .HTTP import HTTP as http
    # and return it
    return http

@foundry(implements=language)
def make():
    """
    The make weaver
    """
    # grab the component class
    from .Make import Make as make
    # and return it
    return make

@foundry(implements=language)
def pfg():
    """
    The pfg weaver
    """
    # grab the component class
    from .PFG import PFG as pfg
    # and return it
    return pfg

@foundry(implements=language)
def perl():
    """
    The perl weaver
    """
    # grab the component class
    from .Perl import Perl as perl
    # and return it
    return perl

@foundry(implements=language)
def python():
    """
    The python weaver
    """
    # grab the component class
    from .Python import Python as python
    # and return it
    return python

@foundry(implements=language)
def sql():
    """
    The SQL weaver
    """
    # grab the component class
    from .SQL import SQL as sql
    # and return it
    return sql

@foundry(implements=language)
def svg():
    """
    The SVG weaver
    """
    # grab the component class
    from .SVG import SVG as svg
    # and return it
    return svg

@foundry(implements=language)
def sh():
    """
    The sh weaver
    """
    # grab the component class
    from .Sh import Sh as sh
    # and return it
    return sh

@foundry(implements=language)
def tex():
    """
    The TeX weaver
    """
    # grab the component class
    from .TeX import TeX as tex
    # and return it
    return tex

@foundry(implements=language)
def xml():
    """
    The XML weaver
    """
    # grab the component class
    from .XML import XML as xml
    # and return it
    return xml


# the templater
def smith(**kwds):
    """
    The templater facility
    """
    # grab the protocol
    from .Smith import Smith as smith
    # build a facility and return it
    return smith(**kwds)

# the protocol that captures the project metadata
from .Project import Project as project

# the templated project implementations
@foundry(implements=project)
def django():
    """
    The django project type
    """
    # grab the component class
    from .Django import Django as django
    # and return it
    return django

@foundry(implements=project)
def plexus():
    """
    The plexus project type
    """
    # grab the component class
    from .Plexus import Plexus as plexus
    # and return it
    return plexus


# end of file
