#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

"""
Read an empty document
"""


def test():
    import pyre.xml
    from pyre.xml.Node import Node
    from pyre.xml.Document import Document


    class Filesystem(Node):
        """The top level document element"""

        def notify(self, parent, locator):
            """do nothing"""

        def __init__(self, parent, attributes, locator):
            """do nothing"""


    class FSD(Document):
        """Document class"""
        # the top-level
        root = "filesystem"
        # declare the handler
        filesystem = pyre.xml.element(tag="filesystem", handler=Filesystem)

    # build the trivial document
    document = FSD()

    # build a parser
    reader = pyre.xml.newReader()
    # parse the sample document
    reader.read(stream=open("sample-empty.xml"), document=document)

    return document


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
