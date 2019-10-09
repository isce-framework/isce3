# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre


# declaration
class Layout(pyre.component):
    """
    A collection of configuration options that define the deployment layout of a pyre application
    """


    # public state
    project = pyre.properties.str(default='pyre')
    project.doc = 'the nickname of this application deployment'

    # the runtime folders
    etc = pyre.properties.uri(default='vfs:/pyre/etc')
    etc.doc = 'the location of application auxiliary data'

    var = pyre.properties.uri(default='vfs:/pyre/var')
    var.doc = 'the location of files that support the application runtime environment'


# end of file
