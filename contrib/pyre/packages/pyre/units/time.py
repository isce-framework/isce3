# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .SI import second
from .SI import pico, nano, micro, milli


#
# definitions of common time units
# data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993

picosecond = pico*second
nanosecond = nano*second
microsecond = micro*second
millisecond = milli*second

# aliases

s = second
ps = picosecond
ns = nanosecond
us = microsecond
ms = millisecond

# other common units

minute = 60 * second
hour = 60 * minute
day = 24 * hour
year = 365.25 * day


# end of file
