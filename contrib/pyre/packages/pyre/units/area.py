# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .length import meter, centimeter, inch, foot, mile


#
# definitions of common area units
# data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993

square_meter = meter**2
square_centimeter = centimeter**2

square_foot = foot**2
square_inch = inch**2
square_mile = mile**2

acre = 43560 * square_foot
hectare = 10000 * square_meter

barn = 1e-28 * square_meter


# end of file
