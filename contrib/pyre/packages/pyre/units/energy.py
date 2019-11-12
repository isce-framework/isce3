# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .SI import joule, kilo, mega, giga, milli


#
# definitions of common energy units
# data taken from
#
#     Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#         fourth edition, John Willey and Sons, 1993
#
#     The NIST Reference on Constants, Units and Uncertainty,
#         http://physics.nist.gov/cuu
#


btu = 1055 * joule
erg = 1e-7 * joule
foot_pound = 1.356 * joule
horse_power_hour = 2.685e6 * joule

calorie = 4.1858 * joule
Calorie = 1000 * calorie
kilowatt_hour = 3.6e6 * joule

electron_volt = 1.60218e-19 * joule


# aliases

J = joule
kJ = kilo*joule
MJ = mega*joule

eV = electron_volt
meV = milli * eV
MeV = mega * eV
GeV = giga * eV

cal = calorie
kcal = kilo*calorie


# end of file
