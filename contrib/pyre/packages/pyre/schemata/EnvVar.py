# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
# superclass
from .String import String


# declaration
class EnvVar(String):
    """
    A type declarator for strings whose default values are associated with an environment variable
    """

    # constants
    typename = 'envvar' # the name of my type


    # meta-methods
    def __init__(self, variable, **kwds):
        # compute the default value by attempting to read the value from the environment
        default = os.environ.get(variable, str())
        # chain up
        super().__init__(default=default, **kwds)
        # save the variable name
        self.envvar = variable
        # all done
        return


# end of file
