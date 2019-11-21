#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the pyre executive can communicate with external command line parsers
"""


def test():
    import pyre.config

    # get the executive instance
    import pyre
    executive = pyre.executive
    # pull the configurator
    configurator = executive.configurator
    # the nameserver
    nameserver = executive.nameserver
    # and a command line parser
    parser = pyre.config.newCommandLineParser()

    # build an argument list
    commandline = [
        '--help',
        '--vtf.nodes=1024',
        '--vtf.(solid,fluid)=solvers',
        '--vtf.(solid,fluid,other).nodes={vtf.nodes}',
        # '--journal.device=file',
        '--journal.debug.main=on',
        '--',
        '--funky-filename',
        'and-a-normal-one'
        ]

    # get the parser to populate the configurator
    events = parser.parse(commandline)
    # and transfer the events to the configurator
    configurator.processEvents(events=events, priority=executive.priority.user)

    # dump the state
    # nameserver.dump()
    # and check that the assignments took place
    assert nameserver["help"] == ""
    assert nameserver["vtf.nodes"] == "1024"
    assert nameserver["vtf.solid"] == "solvers"
    assert nameserver["vtf.fluid"] == "solvers"
    assert nameserver["vtf.solid.nodes"] == "1024"
    assert nameserver["vtf.fluid.nodes"] == "1024"
    # assert nameserver["journal.device"] == "file"
    assert nameserver["journal.debug.main"] == "on"

    # and return the managers
    return executive, parser

# main
if __name__ == "__main__":
    test()


# end of file
