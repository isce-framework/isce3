# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# type conversion from {schemata}
from .. import schemata
# infrastructure from {records}
from .. import records
# structural
field = records.field
derivation = records.derivation
literal = records.literal


# support for sheets
# i have my own measures
from .Measure import Measure as measure
# build the typed descriptors; first the simple ones
bool = measure.bool
decimal = measure.decimal
float = measure.float
inet = measure.inet
int = measure.int
identity = measure.identity
str = measure.str
# next, the more complex types
date = measure.date
dimensional = measure.dimensional
time = measure.time
uri = measure.uri
# finally, containers
list = measure.list
set = measure.set
tuple = measure.tuple

# access to the basic objects in this package
from .Sheet import Sheet as sheet

# dimensions
from .Inferred import Inferred as inferred
from .Interval import Interval as interval
# support for charts
from .Chart import Chart as chart

# reading and writing
# the records class
record = records.record
# the CSV support in records is sufficient for now
csv = records.csv


# the metaclasses
from .Tabulator import Tabulator as tabulator
from .Surveyor import Surveyor as surveyor


# end of file
