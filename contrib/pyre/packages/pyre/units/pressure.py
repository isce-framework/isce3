# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .SI import pascal, kilo, mega, giga

#
# definitions of common pressure units
# data taken from
#     Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#         fourth edition, John Willey and Sons, 1993
#
#     The NIST Reference on Constants, Units and Uncertainty,
#         http://physics.nist.gov/cuu
#


# aliases

Pa = pascal
kPa = kilo*pascal
MPa = mega*pascal
GPa = giga*pascal


# others

bar = 1e5 * pascal
millibar = 100 * pascal

torr = 133.3 * pascal
atmosphere = 101325 * pascal

atm = atmosphere
psi = 6894.757 * pascal


# end of file
