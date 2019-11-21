#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the metaclass correctly builds document and element handlers out of the associated
declarations
"""


# make sure it is processed correctly
def test():
    import pyre.xml
    from pyre.xml.Node import Node
    from pyre.xml.Document import Document

    # the document structure
    class Owner(Node):
        """Handle the owner tag"""

    class Permissions(Node):
        """Handle the permissions tag"""

    class File(Node):
        """Handle the file tag"""
        elements = ("owner", "permissions")

    class Folder(Node):
        """Handle the folder tag"""
        elements = ("owner", "permissions", "file", "folder")

    class Filesystem(Folder):
        """The top level document element"""

    class FSD(Document):
        """Document class"""

        # the top level element
        root = "filesystem"

        # the element descriptors
        owner = pyre.xml.element(tag="owner", handler=Owner)
        permissions = pyre.xml.element(tag="permissions", handler=Permissions)
        file = pyre.xml.element(tag="file", handler=File)
        folder = pyre.xml.element(tag="folder", handler=Folder)
        filesystem = pyre.xml.element(tag="filesystem", handler=Filesystem)


    # verify that the descriptors were correctly harvested by the metaclass
    assert FSD.dtd == ( FSD.owner, FSD.permissions, FSD.file, FSD.folder , FSD.filesystem )

    # verify that each Node has the right (tag, handler constructor) map
    assert Owner._pyre_nodeIndex == {}
    assert Permissions._pyre_nodeIndex == {}
    assert File._pyre_nodeIndex == { "owner": Owner, "permissions": Permissions }
    assert Folder._pyre_nodeIndex == {
        "owner": Owner, "permissions": Permissions, "file": File, "folder": Folder }
    assert Filesystem._pyre_nodeIndex == {
        "owner": Owner, "permissions": Permissions, "file": File, "folder": Folder }

    return FSD


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
