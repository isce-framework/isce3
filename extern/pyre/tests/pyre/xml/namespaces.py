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
    import xml
    import pyre.xml
    from pyre.xml.Node import Node as BaseNode
    from pyre.xml.Document import Document


    class Node(BaseNode):
        """Base class for node handlers of this document"""

        namespace = "http://pyre.orthologue.com/releases/1.0/schema/inventory.html"

    class Inventory(Node):
        """The top level document element"""

        def notify(self, parent, locator):
            """do nothing"""

        def __init__(self, parent, attributes, locator):
            """do nothing"""

    class IDoc(Document):
        """Document class"""
        # the top-level
        root = "inventory"
        # declare the handler
        inventory = pyre.xml.element(tag="inventory", handler=Inventory)


    # build the trivial document
    document = IDoc()

    # build a parser
    reader = pyre.xml.newReader()
    # parse the sample document
    reader.read(
        stream=open("sample-namespaces.xml"),
        document=document,
        features=[(reader.feature_namespaces, True)]
        )

    return document


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
