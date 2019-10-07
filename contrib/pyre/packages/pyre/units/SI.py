# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .Dimensional import Dimensional, dimensionless


# basic SI units
kilogram = Dimensional(value=1.0, derivation=(1, 0, 0, 0, 0, 0, 0))
meter = Dimensional(value=1.0, derivation=(0, 1, 0, 0, 0, 0, 0))
second = Dimensional(value=1.0, derivation=(0, 0, 1, 0, 0, 0, 0))
ampere = Dimensional(value=1.0, derivation=(0, 0, 0, 1, 0, 0, 0))
kelvin = Dimensional(value=1.0, derivation=(0, 0, 0, 0, 1, 0, 0))
mole = Dimensional(value=1.0, derivation=(0, 0, 0, 0, 0, 1, 0))
candela = Dimensional(value=1.0, derivation=(0, 0, 0, 0, 0, 0, 1))

# the standard aliases for these that are used as labels
kg = kilogram
m = meter
s = second
A = ampere
K = kelvin
mol = mole
cd = candela

# the 22 derived SI units with special names
radian = dimensionless                 #  plane angle
steradian = dimensionless              #  solid angle

hertz = 1/second                       #  frequency

newton = meter*kilogram/second**2      #  force
pascal = newton/meter**2               #  pressure
joule = newton*meter                   #  work, heat
watt = joule/second                    #  power, radiant flux

coulomb = ampere*second                #  electric charge
volt = watt/ampere                     #  electric potential difference
farad = coulomb/volt                   #  capacitance
ohm = volt/ampere                      #  electric resistance
siemens = ampere/volt                  #  electric conductance
weber = volt*second                    #  magnetic flux
tesla = weber/meter**2                 #  magnetic flux density
henry = weber/ampere                   #  inductance

celcius = kelvin                       #  Celcius temperature

lumen = candela*steradian              #  luminus flux
lux = lumen/meter**2                   #  illuminance

becquerel = 1/second                   #  radioactivity
gray = joule/kilogram                  #  absorbed dose
sievert = joule/kilogram               #  dose equivalent
katal = mole/second                    #  catalytic activity

# prefixes
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21
yocto = 1e-24

# binary prefixes
# thanks to Elaine Chapin for pointing them out
kibi = 2**10
mebi = kibi**2
gibi = kibi**3
tebi = kibi**4
pebi = kibi**5
exbi = kibi**6


# end of file
