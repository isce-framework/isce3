# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# interface
def zero(entity):
    """
    Zero out the content of {entity}
    """
    return entity.zero()


def fill(entity, value):
    """
    Set all entries in {entity} to {value}
    """
    return entity.fill(value)


# attempt to
try:
    # load the extension module
    from . import gsl
# if this fails
except ImportError:
    # not much to do...
    msg = "could not load the 'gsl' extension module"
    # complain
    import journal
    raise journal.error('gsl').log(msg)

# get the framework
import pyre
# register the package
package = pyre.executive.registerPackage(name='gsl', file=__file__)
# record the layout
home, prefix, defaults = package.layout()

# otherwise, all is well;
# pull in the administrivia
version = gsl.version
copyright = gsl.copyright
def license() : print(gsl.license())


# wrappers
from .Histogram import Histogram as histogram
from .Matrix import Matrix as matrix
from .Permutation import Permutation as permutation
from .RNG import RNG as rng
from .Vector import Vector as vector

# other interfaces
from . import blas, pdf, linalg, stats


# end of file
