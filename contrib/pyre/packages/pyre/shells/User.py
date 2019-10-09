# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
import pyre


# declaration
class User(pyre.component):
    """
    Encapsulation of user specific information
    """


    # configurable state
    # administrative
    name = pyre.properties.str()
    name.doc = 'the full name of the user'

    email = pyre.properties.str()
    email.doc = 'the email address of the user'

    affiliation = pyre.properties.str()
    affiliation.doc = 'the affiliation of the user'

    # choices and defaults
    externals = pyre.externals.dependencies()
    externals.doc = 'the database of preferred instances for each external package category'

    # public data
    uid = os.getuid() # the user's system id
    home = pyre.primitives.path(os.environ.get('HOME')) # the location of the user's home directory
    username = os.environ.get('LOGNAME') # the username


# end of file
