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
    import pyre.xml
    import pyre.filesystem

    class File(pyre.xml.node):
        """Handle the file tag"""

        def notify(self, parent, locator): return

        def __init__(self, parent, attributes, locator):
            self.name = attributes['name']
            self.node = parent.node.folder()
            parent.addEntry(self)


    class Folder(File):
        """Handle the folder tag"""

        elements = ("file", "folder", "metadata")

        def addEntry(self, entry):
            """Add a file to my contents"""
            self.node[entry.name] = entry.node


    class Filesystem(Folder):
        """The top level document element"""

        def notify(self, parent, locator):
            parent.dom = self.node

        def __init__(self, parent, attributes, locator):
            self.node = pyre.filesystem.virtual()


    class FSD(pyre.xml.document):
        """Document class"""

        # the top-level element tag name
        root = "filesystem"

        # the element descriptors
        file = pyre.xml.element(tag="file", handler=File)
        folder = pyre.xml.element(tag="folder", handler=Folder)
        filesystem = pyre.xml.element(tag="filesystem", handler=Filesystem)

        metadata = pyre.xml.element(tag="metadata", handler=pyre.xml.ignorable)


    # build a parser
    reader = pyre.xml.newReader()
    # don't call my handlers on empty element content
    reader.ignoreWhitespace = True

    # parse the sample document
    fs = reader.read(stream=open("sample-fs-extra.xml"), document=FSD())

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
