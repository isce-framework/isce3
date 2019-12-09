# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package contains functions and classes that encapsulate common usage patterns.
"""


# external packages
import itertools, collections


# publish local support
from .Observable import Observable as observable


# utilities
def mean(iterable):
    """
    Compute the mean value of the entries in {iterable}
    """
    # make an iterator over the iterable
    i = iter(iterable)
    # initialize the counters
    count = 1
    total = next(i)
    # loop
    for item in i:
        # update the counters
        count += 1
        total += item
    # all done
    return total * 1/count


def powerset(iterable):
    """
    Compute the full power set, i.e. the set of all permutations, of the given iterable.
    For example:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    Taken from the python itertools documentation
    """
    # convert {iterable} into a list; we need its length
    s = list(iterable)
    # build all possible combinations of all possible lengths
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


# autovivified maps
def vivify(levels=1, atom=dict):
    """
    Builds a nested {defaultdict} with a fixed depth and a specified type at the deepest level.

    Adopted from (http://en.wikipedia.org/wiki/Autovivification)
    """
    # the embryonic case is a {defaultdict} of type {atom}
    if levels < 2: return collections.defaultdict(atom)
    # otherwise, build a {defaultdict} that is shallower by one level
    return collections.defaultdict(lambda: vivify(levels=levels-1, atom=atom))


# factories
def newPathHash(**kwds):
    """
    Build a hashing functor for name hierarchies
    """
    # get the factory
    from .PathHash import PathHash
    # and invoke it
    return PathHash(**kwds)


# cofunctors
from .CoFunctor import CoFunctor as cofunctor
from .Accumulator import Accumulator as accumulator
from .Printer import Printer as printer
from .Tee import Tee as tee


# decorators
def coroutine(f):
    """
    Decorator that automatically primes a coroutine
    """
    # the wrapper
    def wrapper(*args, **kwds):
        # instantiate the generator
        gen = f(*args, **kwds)
        # prime it
        next(gen)
        # and return it
        return gen
    # return the wrapper to the caller
    return wrapper


# end of file
