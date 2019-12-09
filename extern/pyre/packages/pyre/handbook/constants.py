# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

from pyre.units.SI import joule, kelvin, mole

# source: physics.nist.gov/constants
#
# Peter J. Mohr and Barry N. Taylor,
#    CODATA Recommended Values of the Fundamental Physical Constants: 1998
#    Journal of Physical and Chemical Reference Data, to be published
#


avogadro = 6.02214199e23 / mole
boltzmann = 1.3806503e-23 * joule/kelvin

gas_constant = 8.314472 * joule/(mole*kelvin)
gravitational_constant = 6.67408e-11 * meter**3/kilogram/second**2

light_speed = 299792458 * meter/second

# aliases
c = light_speed
k = boltzmann

G = gravitational_constant
N_A = avogadro
R = gas_constant


# end of file
