# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

"""
This package provides support for working with dimensional quantities

It contains definitions for all seven fundamental and twenty-one derived SI units, as well as
support for units from other systems
"""

# convenient access to the local modules
from . import (
    SI,
    angle, area, energy, force, length, mass, power, pressure,
    speed, substance, temperature, time, volume
)


# factories

# this factory grants access to the low level interface, useful for building dimensional
# objects directly from their representation. However, manipulating one of the predefined unit
# objects should be sufficient for most uses. Please let me know if you find something that
# cannot be done any other way and you find yourself resorting to building dimensional
# quantities directly.
from .Dimensional import Dimensional as dimensional, fundamental, zero, one


# the unit parser converts string representations of dimensional quantities into instances of
# Dimensional
from .Parser import Parser as parser


# the parser requires an easy way to load all known unit names
def quantities():
    """
    Build a list of all available modules
    """
    # publish the modules
    return (
        SI,
        angle, area, energy, force, length, mass, power, pressure,
        speed, substance, temperature, time, volume
        )


# end of file
