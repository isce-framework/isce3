# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .SI import kilogram
from .SI import kilo, centi, milli

#
# definitions of common mass units
# data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993
gram = kilogram / kilo
centigram = centi * gram
milligram = milli * gram

# aliases
kg = kilogram
g = gram
cg = centigram
mg = milligram

# other
metric_ton = 1000 * kilogram
ounce = 28.35 * gram
pound = 16 * ounce
lb = pound
ton = 2000 * pound


# end of file
