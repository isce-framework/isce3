#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the framework detects components that do not implement their obligations correctly
"""

# acccess to the parts
import pyre

# declare an interface
class protocol(pyre.protocol):
    """a simple protocol"""
    # properties
    name = pyre.properties.str(default="my name")
    # behaviors
    @pyre.provides
    def say(self):
        """say my name"""

# wrap the component declarations in functions so I can control when the exceptions get raised

def badImplementationSpec():
    class badspec(pyre.component, implements=1):
        """bad implementation specification: not a Protocol subclass"""
    return badspec


def missingProperty():
    class missing(pyre.component, implements=protocol):
        """missing property: doesn't have {name}"""
    # properties
    oops = pyre.properties.str(default="my name")
    # behaviors
    @pyre.export
    def say(self):
        """say my name"""

    return missing


def missingBehavior():
    class missing(pyre.component, implements=protocol):
        """missing behavior: doesn't have {do}"""
    # properties
    name = pyre.properties.str(default="my name")
    # behaviors
    @pyre.export
    def do(self):
        """say my name"""

    return missing


def noExport():
    class missing(pyre.component, implements=protocol):
        """missing behavior decorator"""
    # properties
    name = pyre.properties.str(default="my name")
    # behaviors
    def say(self):
        """say my name"""

    return missing


def test():
    # check that we catch bad implementation specifications
    try:
        badImplementationSpec()
        assert False
    except pyre.components.exceptions.ImplementationSpecificationError as error:
        pass

    # check that we catch missing traits
    try:
        missingProperty()
        assert False
    except pyre.components.exceptions.ProtocolError:
        pass

    # check that we catch missing behaviors
    try:
        missingBehavior()
        assert False
    except pyre.components.exceptions.ProtocolError:
        pass

    # check that we catch missing exports
    try:
        noExport()
        assert False
    except pyre.components.exceptions.ProtocolError:
        pass

    return protocol


# main
if __name__ == "__main__":
    test()


# end of file
