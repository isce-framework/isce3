#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that additional configuration files can be specified on the command line
"""


def test():
    import pyre
    # get the executive instance
    executive = pyre.executive
    # pull the configutor
    cfg = executive.configurator
    # the nameserver
    ns = executive.nameserver
    # and build a command line parser
    parser = executive.newCommandLineParser()
    # build an argument list
    commandline = [
        '--config=sample.pml',
        ]
    # get the parser to populate the configurator
    events = parser.parse(commandline)
    # and transfer the events to the configurator
    cfg.processEvents(events=events, priority=executive.priority.user)
    # now, check that the assignments took place
    assert ns["sample.user.name"] == "michael a.g. aïvázis"
    assert ns["sample.user.email"] == "michael.aivazis@orthologue.com"
    # and return the managers
    return parser


# main
if __name__ == "__main__":
    test()


# end of file
