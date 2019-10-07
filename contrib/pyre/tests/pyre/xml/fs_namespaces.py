#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build a document handler and read a simple file
"""


def test():
    import xml
    import pyre.xml
    import pyre.filesystem
    from pyre.xml.Node import Node as BaseNode
    from pyre.xml.Document import Document


    class Node(BaseNode):
        """Base class for my nodes"""
        namespace = "http://pyre.orthologue.com/releases/1.0/schema/fs.html"

        def notify(self, parent, locator): return

        def __init__(self, parent, attributes, locator):
            self.name = attributes['name']
            self.node = parent.node.folder()
            parent.addEntry(self)


    class File(Node):
        """Handle the file tag"""


    class Folder(Node):
        """Handle the folder tag"""
        elements = ("file", "folder")

        def addEntry(self, entry):
            """Add a file to my contents"""
            self.node[entry.name] = entry.node


    class Filesystem(Folder):
        """The top level document element"""

        def notify(self, parent, locator):
            parent.dom = self.node

        def __init__(self, parent, attributes, locator):
            self.node = pyre.filesystem.virtual()


    class FSD(Document):
        """Document class"""

        # the top-level
        root = "filesystem"

        # the element descriptors
        file = pyre.xml.element(tag="file", handler=File)
        folder = pyre.xml.element(tag="folder", handler=Folder)
        filesystem = pyre.xml.element(tag="filesystem", handler=Filesystem)


    # build a parser
    reader = pyre.xml.newReader()
    # don't call my handlers on empty element content
    reader.ignoreWhitespace = True

    # parse the sample document
    fs = reader.read(
        stream=open("sample-fs-namespaces.xml"),
        document=FSD(),
        features=[(reader.feature_namespaces, True)]
        )

    # dump the contents
    fs.dump(False) # switch to True to see the contents

    # verify
    assert fs is not None
    assert fs["/"] is not None
    assert fs["/tmp"] is not None
    assert fs["/tmp/index.html"] is not None
    assert fs["/tmp/images"] is not None
    assert fs["/tmp/images/logo.png"] is not None

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
