#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that configurations can be populated correctly
"""


def test():
    from pyre.config.events import Assignment
    # build a list to hold the events
    configuration = []
    # create some assignments
    configuration.append(
        Assignment(key=("pyre", "user", "name"), value="michael aïvázis", locator=None))
    configuration.append(
        Assignment(
            key=("pyre", "user", "email"), value="michael.aivazis@orthologue.com",
            locator=None))
    configuration.append(
        Assignment(key=("pyre", "user", "affiliation"), value="orthologue", locator=None))

    # check that they were created and inserted correctly
    assert list(map(str, configuration)) == [
        "{None: ('pyre', 'user', 'name') <- michael aïvázis}",
        "{None: ('pyre', 'user', 'email') <- michael.aivazis@orthologue.com}",
        "{None: ('pyre', 'user', 'affiliation') <- orthologue}",
        ]


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
