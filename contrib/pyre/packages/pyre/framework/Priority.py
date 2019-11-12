# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools # for count
import functools # for total_ordering
import collections # for defaultdict


# auto-implement the ordering methods
@functools.total_ordering
# declaration
class Priority:
    """
    An intelligent enum of configuration priorities

    Each source of configuration information is assigned to a category, and each category has a
    priority. The priority is used to decide whether a the value of a trait should be modified
    based on information from the source at hand: higher priority settings override lower ones.

    Within a category, each configuration setting is assigned a rank, based on the order it was
    encountered. This kind of logic, for example, assures that command line arguments that
    appear later in the command line override earlier ones

    This class provides a resting place for the priority categories, and a total ordering so
    comparisons can be made
    """

    # by default
    category = None

    # the priority category types; patched later in this file
    uninitialized = None
    defaults = None
    boot = None
    package = None
    persistent = None
    user = None
    command = None
    explicit = None
    framework = None


    # meta-methods
    def __init__(self):
        self.rank = next(self.collator[self.category])
        return


    # ordering
    def __eq__(self, other):
        return (self.category, self.rank) == (other.category, other.rank)

    def __lt__(self, other):
        return (self.category, self.rank) < (other.category, other.rank)


    # debug support
    def __str__(self):
        return f"({self.name}:{self.rank})"


    # private data
    collator = collections.defaultdict(itertools.count)
    # narrow the footprint
    __slots__ = ["rank"]


# build a counter
categories = itertools.count(start=-1)


# specific priority categories
class Uninitialized(Priority):
    """
    Category for unspecified priorities; meant to be used as default values for arguments to
    functions
    """
    # public data
    name = 'uninitialized'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Defaults(Priority):
    """
    Category for the priorities of the default values of traits, i.e. the values in the class
    declarations
    """
    # public data
    name = 'defaults'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Boot(Priority):
    """
    Category for the priorities of values assigned while the framework is booting
    """
    # public data
    name = 'boot'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Package(Priority):
    """
    Category for the priorities of values assigned while package configurations are being
    retrieved
    """
    # public data
    name = 'package'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Persistent(Priority):
    """
    Category for the priorities of values retrieved from an application supplied persistent
    store where components record their configurations
    """
    # public data
    name = 'persistent'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class User(Priority):
    """
    Category for the priorities of values assigned during the processing of user configuration
    events
    """
    # public data
    name = 'user'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Command(Priority):
    """
    Category for the priorities of values assigned during the processing of the command line
    """
    # public data
    name = 'command'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Explicit(Priority):
    """
    Category for the priorities of values assigned explicitly by the user program
    """
    # public data
    name = 'explicit'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


class Framework(Priority):
    """
    Category for the priorities of read-only values assigned by the framework
    """
    # public data
    name = 'framework'
    category = next(categories)
    # narrow the footprint
    __slots__ = ()


# patch Priority
Priority.uninitialized = Uninitialized
Priority.defaults = Defaults
Priority.boot = Boot
Priority.package = Package
Priority.command = Command
Priority.user = User
Priority.explicit = Explicit
Priority.framework = Framework


# end of file
