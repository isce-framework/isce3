# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
from pyre import db
# the schema
from . import schema


# connect to the bizbook database under postgres
def pg():
    # make a data store and connect to it
    datastore = db.postgres(name="bizbook").attach()
    # and return it
    return datastore


# connect to the bizbook database under sqlite
def sqlite():
    # make a data store and connect to it
    datastore = db.sqlite(name="bizbook").attach()
    # and return it
    return datastore


# end of file
