# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# markup support
from . import foundry
from . import nexus


# the http service
@foundry(implements=nexus.service)
def http():
    # get the component
    from .http.Server import Server
    # and return it
    return Server


# end of file
