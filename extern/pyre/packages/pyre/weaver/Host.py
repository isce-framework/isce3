# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre
# my protocol
from .Installation import Installation


# class declaration
class Host(pyre.component, family='pyre.smith.installations.host', implements=Installation):
    """
    Encapsulation of the attributes of a remote host for the project templates
    """


    # user configurable state
    name = pyre.properties.str(default='localhost')
    name.doc = "the name of the machine that hosts the live application"

    virtual = pyre.properties.str(default="{project.live.name}")
    virtual.doc = "the virtual name of the web server"

    home = pyre.properties.str(default='~')
    home.doc = "the home directory of the remote user hosting the installation"

    root = pyre.properties.str(default='{project.live.home}/live')
    root.doc = "the home directory of the remote user hosting the installation"

    web = pyre.properties.str(default='{project.live.root}/web')
    web.doc = "the location of web related directories at the remote machine"

    admin = pyre.properties.str(default='root')
    admin.doc = "the username of the remote administrator"


# end of file
