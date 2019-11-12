# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# externals
from .. import descriptors

# access to the descriptor parts
field = descriptors.stem
measure = field.variable
derivation = field.operator
literal = field.literal

# access to the typed field declarators
# basic
bool = descriptors.bool
decimal = descriptors.decimal
float = descriptors.float
inet = descriptors.inet
int = descriptors.int
identity = descriptors.identity
str = descriptors.str
# complex
date = descriptors.date
dimensional = descriptors.dimensional
time = descriptors.time
uri = descriptors.uri

# the decorators
converter = descriptors.converter
normalizer = descriptors.normalizer
validator = descriptors.validator

# the base class for field selectors
from .Selector import Selector as selector
# the record metaclass
from .Templater import Templater as templater
# access to the record class
from .Record import Record as record

# data extraction from formatted streams
from .CSV import CSV as csv


# end of file
