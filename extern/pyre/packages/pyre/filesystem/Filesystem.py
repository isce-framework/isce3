# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref # for {vnodes}, the weak key dictionary
# support
from .. import primitives
# base class
from .Folder import Folder


# declaration
class Filesystem(Folder):
    """
    The base class for representing filesystems

    A filesystem is a special {Folder} that maintains an association between the {Nodes} it
    contains and {Info} objects that are dependent on the specific filesystem type and capture
    what the filesystem knows about them.
    """


    # exceptions
    from .exceptions import NotFoundError, SourceNotFoundError, URISpecificationError, FolderError


    # interface
    def info(self, node):
        """
        Look up and return the available metadata associated with {node}
        """
        # let the exceptions through, for now
        return self.vnodes[node]


    def checksum(self, node, **kwds):
        """
        Compute a checksum for the node
        """
        # i don't know how to do anything smarter
        return id(node)


    def open(self, node, **kwds):
        """
        Open the file associated with {node}
        """
        # i don't know how to do it
        raise NotImplementedError(
            "class {.__name__!r} does not implement 'open'".format(type(self)))


    def discover(self, root=None, **kwds):
        """
        Fill my structure with nodes from an external source
        """
        # N.B.: virtual filesystems do not have a physical entity to query regarding their
        # contents; by definition, they are always fully explored so there is nothing to do

        # deduce the filesystem to which {root} belongs
        fs = root.filesystem() if root is not None else self
        # if i were asked to discover myself or something that belongs to me
        if fs is self:
            # i'm already there...
            return self
        # otherwise, ask the other filesystem to do the work
        return fs.discover(root=root, **kwds)


    # implementation details
    def attach(self, node, uri, metadata=None, **kwds):
        """
        Maintenance for the {vnode} table. Filesystems that maintain more elaborate meta-data
        about their nodes must override to build their {info} structures.
        """
        # if we were not handed any node metadata
        if metadata is None:
            # build a node specific {info} structure
            metadata = node.metadata(uri=uri, **kwds)
        # otherwise
        else:
            # decorate the given structure
            metadata.uri = uri
        # attach it to my vnode table
        self.vnodes[node] = metadata
        # and return it
        return metadata


    # meta methods
    def __init__(self, metadata=None, **kwds):
        # chain up to make me a valid node with me as the filesystem
        super().__init__(filesystem=self, **kwds)
        # my vnode table: a map from nodes to info structures
        self.vnodes = weakref.WeakKeyDictionary()
        # build an info structure for myself
        metadata = self.metadata(uri=primitives.path('/')) if metadata is None else metadata
        # add it to my vnode table
        self.vnodes[self] = metadata
        # all done
        return


# end of file
