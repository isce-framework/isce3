# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref
# my metaclass
from . import _metaclass_Node


# declaration
class Node(metaclass=_metaclass_Node):
    """
    The base class for all filesystem entries

    {Node} and {Folder} are the leaf and container types for the composite that enables the
    representation of the hierarchical structure of filesystems.
    """


    # constants
    isFolder = False


    # types
    # my metadata
    from .InfoFile import InfoFile as metadata
    # exceptions
    from .exceptions import GenericError


    # public data
    @property
    def info(self):
        """
        Return whatever metadata the filesystem maintains about me
        """
        # this much is guaranteed to exist for all well-formed filesystems
        return self.filesystem().info(node=self)


    @property
    def uri(self):
        """
        Return my location relative to the root of my filesystem
        """
        # this much is guaranteed to exist for all well-formed filesystems
        return self.filesystem().info(node=self).uri


    @property
    def marker(self):
        """
        Return my distinguishing mark used by explorers to decorate their reports
        """
        # this much is guaranteed to exist for all well-formed filesystems
        return self.filesystem().info(node=self).marker


    # interface
    def checksum(self, **kwds):
        """
        Build a checksum that summarizes my contents
        """
        # delegate to the filesystem
        return self.filesystem().checksum(node=self, **kwds)


    def open(self, **kwds):
        """
        Access the contents of the physical resource with which I am associated
        """
        # delegate the action to the filesystem
        return self.filesystem().open(node=self, **kwds)


    # meta methods
    def __init__(self, filesystem, **kwds):
        """
        Build a node that is contained in the given {filesystem}
        """
        # chain up
        super().__init__(**kwds)
        # build a weak reference to my filesystem
        self.filesystem = weakref.ref(filesystem)
        # and return
        return


    # debugging support
    def dump(self, indent=0):
        """
        Print out my contents using a tree explorer
        """
        # grab the tree explorer factory
        from . import treeExplorer
        # build one
        explorer = treeExplorer(indent=indent)
        # get the representation of my contents and dump it out
        yield from explorer.explore(node=self, label=str(self.uri))
        # all done
        return


    def print(self):
        """
        Display my contents
        """
        # easy...
        return print('\n'.join(self.dump()))


# end of file
