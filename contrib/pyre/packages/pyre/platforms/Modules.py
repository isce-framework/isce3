# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# framework
import pyre
# superclass
from .Managed import Managed


# declaration
class Modules(Managed, family='pyre.platforms.packagers.modules'):
    """
    Support for the modules package manager
    """


    # public data
    name = 'modules'
    client = 'modulecmd'


    # protocol obligations
    @pyre.export
    def prefix(self):
        """
        Retrieve the package manager install location
        """
        # check my cache
        prefix = self._prefix
        # for whether I have done this successfully before
        if prefix:
            # in which case we are done
            return prefix

        # otherwise, attempt to
        try:
            # look for the magic environment variable
            prefix = os.environ['MODULESHOME']
        # if it's not there
        except KeyError:
            # this is not a modules machine
            msg = 'could not locate {.name!r}'.format(self)
            # so complain
            raise self.ConfigurationError(configurable=self, errors=[msg])

        # found it; convert to a path
        prefix = pyre.primitives.path(prefix)

        # save the prefix
        self._prefix = prefix
        # build the path to the command
        self.client = prefix / 'bin' / 'modulecmd'

        # return the prefix to the client
        return prefix


    # private data
    _prefix = None


# end of file
