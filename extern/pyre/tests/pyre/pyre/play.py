#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access the framework
import pyre


class role(pyre.protocol, family='play.roles'):
    """A role"""
    @classmethod
    def pyre_default(cls):
        return actor


class actor(pyre.component, implements=role):
    """An actor"""

# protect this class declaration because the configuration file has an intentional error
def musical(**kwds):
    # build the embedded class record
    class musical(pyre.component, family='play.musicals'):
        """A play"""

        cast = pyre.properties.dict(schema=role())

    # return it
    return musical(**kwds)


# driver
def test():

    # attempt to
    try:
        # make a play
        musical(name='spamalot')
    # if this fails
    except role.ResolutionError:
        # all good
        pass
    # otherwise
    else:
        # oops
        assert False

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
