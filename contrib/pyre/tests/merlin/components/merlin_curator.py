#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the curator can save and load project files
"""


def test():
    # access to the merlin executive
    from merlin import merlin

    # get the curator
    curator = merlin.curator
    # create a project description object
    project = merlin.newProject(name="test")
    # archive it
    curator.saveProject(project)
    # refresh the folder
    merlin.pfs['project'].discover()
    # and load it back in
    project = curator.loadProject()

    # check the name
    assert project.name == "test"

    # and return
    return project


# main
if __name__ == "__main__":
    test()


# end of file
