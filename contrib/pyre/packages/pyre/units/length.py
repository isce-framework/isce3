# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .SI import meter
from .SI import kilo, centi, milli, micro, nano


#
# definitions of common length units
# data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993

nanometer = nano * meter
micrometer = micro * meter
millimeter = milli * meter
centimeter = centi * meter
kilometer = kilo * meter


# aliases

m = meter
nm = nanometer
um = micrometer
micron = micrometer
mm = millimeter
cm = centimeter
km = kilometer


# British units

inch = 2.540 * centimeter
foot = 12 * inch
yard = 3 * foot
mile = 5280 * foot

mil = 1e-3 * inch
fathom = 6 * foot
nautical_mile = 1852 * meter

# others

angstrom = 1e-10 * meter
fermi = 1e-15 * meter

astronomical_unit = 1.49598e11 * meter
light_year = 9.460e12 * kilometer
parsec = 3.084e13 * kilometer


# end of file
