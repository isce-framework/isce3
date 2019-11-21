#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise loading settings from configuration files
"""


def test():
    import pyre

    # get the pyre executive
    executive = pyre.executive
    # gain access to the configurator
    c = executive.configurator
    # and the nameserver
    ns = executive.nameserver

    # attempt to request a node that doesn't exist yet
    try:
        ns["sample.user.name"]
        assert False
    except ns.UnresolvedNodeError:
        pass

    # this is the node that we build out of the configuration
    try:
        ns["sample.user.byline"]
        assert False
    except ns.UnresolvedNodeError:
        pass
    # so define it
    ns["sample.user.byline"] = "{sample.user.name} -- {sample.user.email}"

    # load a configuration file
    pyre.loadConfiguration("sample.pfg")

    # report any errors
    errors = executive.errors
    if errors and executive.verbose:
        count = len(errors)
        s = '' if count == 1 else 's'
        print(' ** during configuration: {} error{}:'.format(len(errors), s))
        for error in errors:
            print(' -- {}'.format(error))
    # ns.dump()

    # try again
    assert ns["sample.user.name"] == "michael a.g. aïvázis"
    assert ns["sample.user.email"] == "michael.aivazis@orthologue.com"
    assert ns["sample.user.affiliation"] == "orthologue"
    # and the local one
    assert ns["sample.user.byline"] == "michael a.g. aïvázis -- michael.aivazis@orthologue.com"

    # make a change
    ns["sample.user.affiliation"] = "orthologue"
    ns["sample.user.email"] = "michael.aivazis@orthologue.com"
    # check
    assert ns["sample.user.affiliation"] == "orthologue"
    assert ns["sample.user.byline"] == "michael a.g. aïvázis -- michael.aivazis@orthologue.com"

    # all good
    return executive


# main
if __name__ == "__main__":
    test()


# end of file
