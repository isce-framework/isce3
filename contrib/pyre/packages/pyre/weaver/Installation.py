# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre


# declaration
class Installation(pyre.protocol, family='pyre.smith.installations'):
    """
    Encapsulation of the configuration and layout of the machine hosting the installation
    """


    # user configurable state
    name = pyre.properties.str()
    name.doc = "the name of the machine that hosts the live application"

    virtual = pyre.properties.str()
    virtual.doc = "the virtual name of the web server"

    home = pyre.properties.str()
    home.doc = "the home directory of the remote user hosting the installation"

    root = pyre.properties.str()
    root.doc = "the installation directory on the hosting machine"

    web = pyre.properties.str()
    web.doc = "the location of web related directories at the hosting machine"

    admin = pyre.properties.str()
    admin.doc = "the username of the remote administrator"


    # framework obligations
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Build the preferred host implementation
        """
        # the default installation is a posix host
        from .Host import Host
        return Host


# end of file
