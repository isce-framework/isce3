# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os, glob
# access the pyre framework
import pyre


# protocol declaration
class Package(pyre.protocol, family='pyre.externals'):
    """
    The protocol that all external package managers must implement
    """

    # configurable state
    version = pyre.properties.str(default="unknown")
    version.doc = 'the package version'

    prefix = pyre.properties.path()
    prefix.doc = 'the package installation directory'

    # constants
    category = None # the common name for this package category


    # framework support
    @classmethod
    def pyre_default(cls, channel=None, **kwds):
        """
        Identify the default implementation of a package
        """
        # get the user
        user = cls.pyre_user
        # check whether there is a registered preference for this category
        try:
            # if so, we are done
            return user.externals[cls.category]
        # if not
        except (KeyError, AttributeError):
            # moving on
            pass

        # next, get the host
        host = cls.pyre_host
        # check whether there is a registered preference for this category
        try:
            # if so, we are done
            return host.externals[cls.category]
        # if not
        except (KeyError, AttributeError):
            # moving on
            pass

        # finally, get the package manager
        packager = host.packager
        # go through my host specific choices
        for package in packager.packages(category=cls):
            # i only care about the first one
            return package

        # if i get this far, no one knows what to do
        return
        raise cls.DefaultError(protocol=cls)


# end of file
