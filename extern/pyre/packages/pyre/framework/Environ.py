# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
import weakref
import collections.abc
from .. import tracking


# declaration
class Environ(collections.abc.MutableMapping):
    """
    """

    # constants
    prefix = 'pyre.environ.'


    # meta-methods
    def __init__(self, executive, **kwds):
        # chain up
        super().__init__(**kwds)
        # i need the nameserver
        nameserver = executive.nameserver
        # save a weak reference to it
        self.nameserver = weakref.proxy(nameserver)

        # build a locator
        locator = tracking.simple('while booting')
        # and pick a priority
        priority = nameserver.priority.boot

        # get the name and value of every environment variable
        for name, value in os.environ.items():
            # attempt to
            try:
                # make a node in the configuration store
                nameserver.configurable(
                    name=self.prefix+name, configurable=value,
                    locator=locator, priority=priority())
            # if anything goes wrong
            except nameserver.FrameworkError:
                # just skip this variable for now
                continue

        # all done
        return


    def __setitem__(self, key, value):
        # place the value in the environment
        os.environ[key] = value

        # build a locator
        locator = tracking.here(-1)
        # a priority
        priority = self.nameserver.priority.user()
        # and add it to the configuration store
        self.nameserver.configurable(
            name=self.prefix+key, value=value, locator=locator, priority=priority)

        # all done
        return


    def __getitem__(self, key):
        # don't mess with this; just get it directly from the environment
        return os.environ[key]


    def __delitem__(self, key):
        # remove the variable from the environment
        del os.environ[key]
        # currently, there is no way to delete configuration settings, so we are done
        return


    def __iter__(self):
        # set up an iteration over the environment variables
        return iter(os.environ)


    def __len__(self):
        # ask the environment
        return len(os.environ)


    # private data
    nameserver = None


# end of file
